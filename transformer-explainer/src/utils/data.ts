import {
	modelData,
	tokens,
	tokenIds,
	isModelRunning,
	predictedToken,
	traceServerUrl,
	tracePair,
	cramEnabled,
	cramSpan,
	inspection,
	activeTrace
} from '~/store';
import { showFlowAnimation } from './animation';

// ---------------------------------------------------------------------------
// Qwen trace mode: fetch a vanilla+CrAM activation trace from the Python backend
// and map it into the ModelData shape the components already consume.
// ---------------------------------------------------------------------------
export const traceToModelData = (trace: any): ModelData => {
	const T = trace.tokens.length;
	const outputs: ModelData['outputs'] = {};
	trace.layers.forEach((layer: any, i: number) => {
		layer.heads.forEach((h: any, j: number) => {
			const base = `block_${i}_attn_head_${j}`;
			const put = (suffix: string, mat: number[][]) => {
				outputs[`${base}_${suffix}`] = { data: mat, dims: [1, 1, T, T], size: T * T };
			};
			put('attn', h.scores);
			put('attn_scaled', h.scaled);
			put('attn_masked', h.masked);
			put('attn_softmax', h.softmax);
			put('attn_dropout', h.softmax); // dropout disabled during capture
		});
	});
	const probabilities: Probabilities = trace.next_token.map((t: any, rank: number) => ({
		rank,
		tokenId: t.id,
		token: formatTokenForDisplay(t.token),
		logit: 0,
		scaledLogit: 0,
		expLogit: 0,
		probability: t.prob
	}));
	return { logits: [], outputs, probabilities, sampled: probabilities[0] };
};

export const runModelFromTrace = async ({
	input,
	span = null,
	credibility = 0,
	useCram = false,
	maxTokens = 16
}: {
	input: string;
	span?: [number, number] | null;
	credibility?: number;
	useCram?: boolean;
	maxTokens?: number;
}) => {
	isModelRunning.set(true);
	try {
		const res = await fetch(`${traceServerUrl}/api/trace`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text: input || ' ', span, credibility, max_tokens: maxTokens })
		});
		if (!res.ok) throw new Error(`trace server ${res.status}`);
		const payload = await res.json();
		tracePair.set(payload);
		const trace = useCram && payload.cram ? payload.cram : payload.vanilla;
		const md = traceToModelData(trace);
		activeTrace.set(trace);
		tokens.set(trace.tokens);
		tokenIds.set(trace.token_ids);
		modelData.set(md);

		setTimeout(async () => {
			await showFlowAnimation(trace.tokens.length, false);
			predictedToken.set(md.sampled);
			isModelRunning.set(false);
		}, 0);
		return payload;
	} catch (err) {
		console.error('runModelFromTrace failed:', err);
		isModelRunning.set(false);
		throw err;
	}
};

export const loadAudienceInspection = async (qid: string, claimId: string) => {
	isModelRunning.set(true);
	try {
		const res = await fetch(
			`${traceServerUrl}/api/audience/inspect/${encodeURIComponent(qid)}/${encodeURIComponent(claimId)}`
		);
		if (!res.ok) throw new Error(`inspection server ${res.status}`);
		const payload = await res.json();
		tracePair.set(payload);
		inspection.set(payload.inspection);
		cramSpan.set(payload.inspection.display_span);
		cramEnabled.set(false);
		applyTracePair(payload, false);
		return payload;
	} finally {
		isModelRunning.set(false);
	}
};

// Re-render an already-fetched trace pair when the CrAM toggle flips (no refetch).
export const applyTracePair = (payload: any, useCram: boolean) => {
	const trace = useCram && payload?.cram ? payload.cram : payload?.vanilla;
	if (!trace) return;
	const md = traceToModelData(trace);
	activeTrace.set(trace);
	tokens.set(trace.tokens);
	tokenIds.set(trace.token_ids);
	modelData.set(md);
	predictedToken.set(md.sampled);
};

// Map a real (signed, unbounded) Qwen vector to [0,1] for the VectorCanvas color
// scale. Symmetric min/max so 0 sits in the middle and the pattern is preserved.
export const normalizeVector = (vec: number[]): number[] => {
	if (!vec || vec.length === 0) return [];
	let m = 0;
	for (const v of vec) m = Math.max(m, Math.abs(v));
	if (m === 0) return vec.map(() => 0.5);
	return vec.map((v) => 0.5 + (0.5 * v) / m);
};

export const adjustTemperature = async () => {
	// Qwen traces already contain probabilities from the backend forward pass.
};

// Helper function to format tokens for display
function formatTokenForDisplay(token: string): string {
	// Replace special whitespace characters with readable labels
	return token
		.replace(/\n/g, '[NEWLINE]')
		.replace(/\t/g, '[TAB]')
		.replace(/\r/g, '[CR]')
		.replace(/\s{2,}/g, (match) => `[${match.length} SPACES]`); // Multiple spaces
}
