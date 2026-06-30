export type TokenWordGroup = {
	label: string;
	start: number;
	end: number;
};

// Reconstruct display words from Qwen's subword tokens. Tensor indices stay
// token-level; this changes labels and selection spans only.
export function groupTokens(tokens: string[], sourceIndices?: number[]): TokenWordGroup[] {
	const groups: TokenWordGroup[] = [];

	for (let i = 0; i < tokens.length; i++) {
		const raw = tokens[i] ?? '';
		const sourceBreak =
			i > 0 &&
			sourceIndices?.[i] != null &&
			sourceIndices?.[i - 1] != null &&
			sourceIndices[i] !== sourceIndices[i - 1] + 1;
		const startsWord = i === 0 || sourceBreak || /^\s/.test(raw);

		if (startsWord) {
			groups.push({ label: raw.trim(), start: i, end: i + 1 });
		} else {
			const group = groups[groups.length - 1];
			group.label += raw;
			group.end = i + 1;
		}
	}

	return groups.filter((group) => group.label.length > 0);
}

export function tokenGroupLabels(tokens: string[], sourceIndices?: number[]) {
	const labels = tokens.map(() => ({ label: '', span: 1, continuation: true }));
	for (const group of groupTokens(tokens, sourceIndices)) {
		labels[group.start] = {
			label: group.label,
			span: group.end - group.start,
			continuation: false
		};
	}
	return labels;
}
