<script lang="ts">
	import {
		tokens,
		cramEnabled,
		cramSpan,
		cramCredibility,
		tracePair,
		inputText,
		inspection,
		activeTrace
	} from '~/store';
	import { runModelFromTrace, applyTracePair } from '~/utils/data';
	import { groupTokens, type TokenWordGroup } from '~/utils/tokenGroups';
	import { get } from 'svelte/store';
	import Katex from '~/utils/Katex.svelte';

	let selected = -1; // token index to down-weight (-1 = none)
	let selectedEnd = 0;
	let credibility = 0; // 0 = fully suppress
	let busy = false;
	let minimized = false;
	let inspectionMode = false;
	$: inspectionMode = !!$inspection;
	$: wordGroups = groupTokens($tokens, $activeTrace?.source_indices);
	$: selectedHeadMap = $tracePair?.cram?.cram?.selected_heads || {};
	$: selectedHeadCount = Object.values(selectedHeadMap).reduce(
		(total: number, heads: any) => total + (Array.isArray(heads) ? heads.length : 0),
		0
	);
	$: if ($inspection?.display_span) selected = $inspection.display_span[0];
	$: if ($inspection?.display_span) selectedEnd = $inspection.display_span[1];

	// reset the picked token if the input (and thus tokens) changed
	$: if ($tokens && selected >= $tokens.length) selected = -1;

	async function fetchPair() {
		if (selected < 0) return;
		if (inspectionMode) return;
		busy = true;
		cramSpan.set([selected, selectedEnd]);
		cramCredibility.set(credibility);
		try {
			await runModelFromTrace({
				input: get(inputText).trim(),
				span: [selected, selectedEnd],
				credibility,
				useCram: get(cramEnabled),
				maxTokens: 16
			});
		} catch (e) {
			console.error(e);
		} finally {
			busy = false;
		}
	}

	function setView(on: boolean) {
		cramEnabled.set(on);
		const pair = get(tracePair);
		if (pair?.cram && selected >= 0) applyTracePair(pair, on);
		else if (on && selected >= 0) fetchPair();
	}

	function pick(group: TokenWordGroup) {
		if (inspectionMode) return;
		const wasSelected = selected === group.start && selectedEnd === group.end;
		selected = wasSelected ? -1 : group.start;
		selectedEnd = wasSelected ? 0 : group.end;
		cramSpan.set(selected >= 0 ? [selected, selectedEnd] : null);
		if (selected >= 0) fetchPair();
		else setView(false);
	}

	const label = (t: string) => (t.trim() === '' ? '␣' : t.trim());
</script>

<div class="cram-panel" class:minimized>
	<div class="cram-header">
		<div class="cram-title">
			CrAM — credibility-aware attention
			{#if selectedHeadCount}<span>· {selectedHeadCount} influential heads</span>{/if}
		</div>
		<button
			type="button"
			class="minimize"
			aria-label={minimized ? 'Expand CrAM controls' : 'Minimize CrAM controls'}
			title={minimized ? 'Expand' : 'Minimize'}
			on:click={() => (minimized = !minimized)}
		>
			{minimized ? '＋' : '−'}
		</button>
	</div>

	{#if !minimized}
		<div class="cram-row">
			<span class="cram-label">Down-weight word</span>
			<div class="chips">
				{#each wordGroups as group}
					<button
						class="chip"
						class:active={selected < group.end && selectedEnd > group.start}
						disabled={inspectionMode}
						title={`${group.end - group.start} Qwen token(s)`}
						on:click={() => pick(group)}>{label(group.label)}</button
					>
				{/each}
			</div>
		</div>

		<div class="cram-row">
			<span class="cram-label">Credibility</span>
			<input
				type="range"
				min="0"
				max="1"
				step="0.05"
				bind:value={credibility}
				on:change={fetchPair}
				disabled={inspectionMode}
				aria-label="Token credibility"
			/>
			<span class="cred-val">{credibility.toFixed(2)}</span>
		</div>

		<div class="cram-row view">
			<button class="seg" class:active={!$cramEnabled} on:click={() => setView(false)}>Vanilla</button>
			<button class="seg" class:active={$cramEnabled} disabled={selected < 0} on:click={() => setView(true)}>
				CrAM ↓
			</button>
			{#if busy}<span class="busy">computing…</span>{/if}
		</div>

		{#if $cramEnabled && selected >= 0}
			<div class="cram-formula">
				<Katex
					math={'A_h^{\\mathrm{CrAM}}=\\operatorname{Norm}_1(A_h\\odot\\bar{s})'}
				/>
				<div>
					s̄ is normalized token credibility. For s̄ &gt; 0 this is softmax(Z<sub>h</sub> + log s̄);
					s̄ = 0 hard-masks the token. Vanilla uses A<sub>h</sub> = softmax(Z<sub>h</sub>).
				</div>
			</div>
		{/if}

		{#if inspectionMode}
			<div class="inspection">
				<div><span>Question</span>{$inspection.question}</div>
				<div><span>Truth</span>{$inspection.truth}</div>
				<div><span>Fake passage</span>{$inspection.claim}</div>
				<div class="answers">
					<span>Vanilla: <b>{$inspection.vanilla_answer}</b></span>
					<span>CrAM: <b>{$inspection.cram_answer}</b></span>
				</div>
			</div>
			<div class="hint">Highlighted tokens belong to the fake passage. Switch Vanilla ↔ CrAM to compare.</div>
		{:else if selected < 0}
			<div class="hint">Set credibility, pick a word above, then switch to CrAM.</div>
		{:else}
			<div class="hint">
				Down-weighting <b>"{wordGroups.find((group) => group.start === selected)?.label || label($tokens[selected])}"</b>
				(credibility {credibility.toFixed(2)}). Switch Vanilla ↔ CrAM to compare.
			</div>
		{/if}
	{/if}
</div>

<style>
	.cram-panel {
		position: fixed;
		left: 16px;
		bottom: 16px;
		z-index: 50;
		width: 340px;
		max-width: calc(100vw - 32px);
		max-height: calc(100vh - 32px);
		overflow: auto;
		padding: 12px 14px;
		border-radius: 10px;
		background: #0d1024;
		border: 1px solid #6366f1aa;
		box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
		color: #e2e8f0;
		font-size: 12px;
	}
	.cram-panel.minimized {
		width: auto;
		overflow: hidden;
	}
	.cram-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
	}
	.cram-title {
		font-weight: 700;
		font-size: 12px;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: #a78bfa;
		margin-bottom: 10px;
	}
	.cram-title span {
		color: #64748b;
		font-size: 10px;
		letter-spacing: 0;
		white-space: nowrap;
	}
	.minimized .cram-title {
		margin-bottom: 0;
	}
	.minimize {
		flex: none;
		width: 24px;
		height: 24px;
		border-radius: 6px;
		border: 1px solid #4c5680;
		background: #171a32;
		color: #c4b5fd;
		font-size: 16px;
		line-height: 1;
		cursor: pointer;
	}
	.cram-row {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 8px;
	}
	.cram-label {
		flex: none;
		width: 96px;
		color: #94a3b8;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.chip {
		padding: 2px 7px;
		border-radius: 999px;
		border: 1px solid #26304f;
		background: #111428;
		color: #cbd5e1;
		cursor: pointer;
		font-family: ui-monospace, monospace;
		font-size: 11px;
	}
	.chip.active {
		border-color: #f87171;
		background: #f8717122;
		color: #fecaca;
	}
	.chip:disabled {
		cursor: default;
	}
	input[type='range'] {
		flex: 1;
	}
	.cred-val {
		font-family: ui-monospace, monospace;
		color: #a78bfa;
		width: 34px;
		text-align: right;
	}
	.view .seg {
		flex: 1;
		padding: 6px;
		border-radius: 7px;
		border: 1px solid #26304f;
		background: #111428;
		color: #cbd5e1;
		cursor: pointer;
		font-weight: 600;
	}
	.view .seg.active {
		border-color: #a78bfa;
		background: #a78bfa22;
		color: #ddd6fe;
	}
	.view .seg:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.busy {
		color: #fbbf24;
		flex: none;
	}
	.hint {
		margin-top: 8px;
		color: #64748b;
		line-height: 1.4;
	}
	.cram-formula {
		margin: 8px 0;
		padding: 7px 8px;
		border: 1px solid #4c1d95;
		border-radius: 7px;
		background: #17132d;
		color: #ddd6fe;
		overflow-x: auto;
	}
	.cram-formula div {
		margin-top: 4px;
		color: #94a3b8;
		font-size: 10px;
	}
	.hint b {
		color: #fecaca;
	}
	.inspection {
		display: grid;
		gap: 5px;
		margin-top: 8px;
		padding-top: 8px;
		border-top: 1px solid #26304f;
		line-height: 1.35;
	}
	.inspection div > span {
		display: inline-block;
		width: 72px;
		color: #94a3b8;
	}
	.inspection .answers {
		display: flex;
		gap: 12px;
	}
	.inspection .answers > span {
		width: auto;
	}
	.inspection .answers b {
		color: #ddd6fe;
	}
</style>
