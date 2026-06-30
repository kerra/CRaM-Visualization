<script lang="ts">
	import {
		tokens,
		expandedBlock,
		vectorHeight,
		inputText,
		rootRem,
		sampling,
		maxVectorHeight,
		minVectorHeight,
		maxVectorScale,
		headContentHeight,
		temperature,
		modelData,
		isFetchingModel,
		isOnBlockTransition,
		blockIdx,
		cramEnabled,
		cramSpan,
		inspection
	} from '~/store';
	import Sankey from '~/components/Sankey.svelte';
	import Attention from '~/components/Attention.svelte';
	import SubsequentBlocks from '~/components/SubsequentBlocks.svelte';
	import LinearSoftmax from '~/components/LinearSoftmax.svelte';
	import Embedding from '~/components/Embedding.svelte';
	import Mlp from '~/components/Mlp.svelte';

	import { onMount } from 'svelte';
	import classNames from 'classnames';

	import { adjustTemperature, runModelFromTrace, loadAudienceInspection } from '~/utils/data';
	import WeightPopovers from '~/components/WeightPopovers.svelte';
	import { fade } from 'svelte/transition';
	import BlockTransition from '~/components/BlockTransition.svelte';
	import QKV from '~/components/QKV.svelte';
	import CramControl from '~/components/CramControl.svelte';

	let active = false;
	let viewportWidth = 1300;
	let viewportHeight = 720;
	const visualizationWidth = 1300;
	$: minimumTokenHeight =
		titleHeight +
		rootRem * 5 +
		Math.max(1, $tokens.length) * minVectorHeight * maxVectorScale +
		rootRem * 0.5 * Math.max(0, $tokens.length - 1);
	$: availableVisualizationHeight = Math.max(280, viewportHeight - rootRem * 5);
	$: visualizationScale = Math.min(
		1,
		viewportWidth / visualizationWidth,
		availableVisualizationHeight / minimumTokenHeight
	);

	// Qwen trace mode: no in-browser model. We fetch activation traces from the
	// Python backend (trace_backend/server.py on :8200) instead of running ONNX.
	onMount(() => {
		active = true;
		isFetchingModel.set(false);
		let unsubscribe = () => {};
		const start = async () => {
			const params = new URLSearchParams(window.location.search);
			const qid = params.get('qid');
			const claim = params.get('claim');
			if (qid && claim) {
				try {
					const payload = await loadAudienceInspection(qid, claim);
					inputText.set(payload.inspection.claim);
					unsubscribe = subscribeInputs(true);
					return;
				} catch (err) {
					console.error('Could not load audience inspection:', err);
				}
			}
			unsubscribe = subscribeInputs();
		};
		start();
		return () => unsubscribe();
	});

	// Subscribe inputs
	const subscribeInputs = (skipInitial = false) => {
		let firstInput = true;
		const runModelOrCache = () => {
			if (skipInitial && firstInput) {
				firstInput = false;
				return;
			}
			firstInput = false;
			inspection.set(null);
			cramEnabled.set(false);
			cramSpan.set(null);
			// Qwen trace mode: always fetch the activation trace from the backend.
			runModelFromTrace({ input: $inputText.trim() });
		};

		const unsubscribeInputText = inputText.subscribe((value) => {
			runModelOrCache();
		});

		let initialTemperature = true; // prevent initial redundant rendering
		const unsubscribeTemperature = temperature.subscribe((value) => {
			if (initialTemperature) {
				initialTemperature = false;
				return;
			}
			adjustTemperature();
		});

		let initialSampling = true; // prevent initial redundant rendering
		const unsubscribeSmapling = sampling.subscribe((value) => {
			if (initialSampling) {
				initialSampling = false;
				return;
			}
			adjustTemperature();
		});

		return () => {
			unsubscribeInputText();
			unsubscribeTemperature();
			unsubscribeSmapling();
		};
	};

	// visual elements
	let vizHeight = 0;
	let titleHeight = rootRem * 5;

	const calculateVectorHeight = () => {
		const gaps = rootRem * 0.5 * ($tokens.length - 1);
		const vectorHeightVal = Math.min(
			Math.max((vizHeight - titleHeight - gaps) / $tokens.length / maxVectorScale, minVectorHeight),
			maxVectorHeight
		);
		vectorHeight.set(vectorHeightVal);
		headContentHeight.set(Math.max($tokens.length * vectorHeightVal * 3 + gaps, rootRem * 20));
	};

	$: if (vizHeight || $tokens.length) {
		calculateVectorHeight();
	}
</script>

<svelte:window bind:innerWidth={viewportWidth} bind:innerHeight={viewportHeight} />

<div
	class:active
	class="main-section h-full w-full"
	style={`--vector-height: ${$vectorHeight}px;--title-height: ${titleHeight}px;--content-height:${vizHeight - titleHeight}px;`}
>
	{#if !!$expandedBlock.id}
		<div
			class={classNames('dim', `${$expandedBlock.id || ''}`)}
			transition:fade={{ duration: 100 }}
		></div>
		<div
			class={classNames('dim-partial left', `${$expandedBlock.id || ''}`)}
			transition:fade={{ duration: 100 }}
		></div>
		<div
			class={classNames('dim-partial right', `${$expandedBlock.id || ''}`)}
			transition:fade={{ duration: 100 }}
		></div>
	{/if}
	<div class="sankey opacity-1" class:attention={$expandedBlock.id === 'attention'}>
		<Sankey />
	</div>
	<CramControl />
	<div
		class="nodes resize-watch"
		style={`width:${100 / visualizationScale}%;height:${100 / visualizationScale}%;transform:scale(${visualizationScale});`}
	>
		<div class="steps" class:expanded={!!$expandedBlock.id} bind:offsetHeight={vizHeight}>
			<Embedding className="step" />
			<div class="blocks relative">
				<div class="block-steps main" class:initial={$blockIdx === 0}>
					<QKV className="step" />
					<Attention className="step" />
					<Mlp className="step" />
				</div>
				<div
					class="block-steps next"
					class:hide={!$isOnBlockTransition}
					class:initial={$blockIdx === 0}
				>
					<QKV className="step" />
					<Attention className="step" />
					<Mlp className="step" />
				</div>
				<div class="transition-watch" class:hide={!$isOnBlockTransition}></div>
			</div>
			<SubsequentBlocks className="step" />
			<LinearSoftmax className="step" />
		</div>
		<WeightPopovers />
		<BlockTransition />
	</div>
</div>

<style lang="scss">
	.main-section {
		opacity: 0;
		&.active {
			opacity: 1;
		}
	}
	.nodes {
		height: 100%;
		width: 100%;
		padding: 1rem 0 3rem 0;
		position: relative;
		transform-origin: top left;
	}
	.steps {
		position: relative;
		width: 100%;
		height: 100%;
		position: relative;
		display: grid;
		grid-template-columns: auto 3.5fr 0.5fr 0.5fr;

		&.expanded {
			:global(.step > .title) {
				padding-bottom: 3rem;
			}
		}

		.blocks {
			position: relative;
			width: 100%;
			height: 100%;

			.block-steps {
				height: 100%;
				width: 100%;
				position: absolute;
				display: grid;
				grid-template-columns: 0.5fr 2fr 1fr;
			}
			.block-steps.main {
				transform-origin: 3rem center;
				top: 0;
				left: 0;
			}
			.block-steps.next {
				transform-origin: right center;
				justify-content: end;
				top: 0;
				right: 0;
				pointer-events: none;
			}

			.transition-watch {
				position: absolute;
				top: 0;
				left: 0;
				height: 100%;
				width: 100%;
				pointer-events: none;
			}

			.hide {
				display: none;
			}
			&.animate-forward {
				.block-steps,
				.transition-watch {
					animation-duration: 800ms;
					animation-timing-function: ease-in;
				}
				.block-steps.main {
					animation-name: collapse;
					&.initial {
						transform-origin: left center;
					}
				}
				.block-steps.next {
					animation-name: expand;
				}
				.transition-watch {
					animation-name: width-collapse;
				}
			}

			&.animate-backward {
				.block-steps,
				.transition-watch {
					animation-duration: 800ms;
					animation-timing-function: ease-in;
				}
				.block-steps.main {
					animation-name: expand;
					&.initial {
						transform-origin: left center;
					}
				}
				.block-steps.next {
					animation-name: collapse;
				}
				.transition-watch {
					animation-name: width-collapse;
				}
			}
		}
	}
	@keyframes width-collapse {
		0% {
			width: 100%;
		}
		100% {
			width: 0%;
		}
	}
	@keyframes expand {
		0% {
			transform: scaleX(0);
		}
		100% {
			transform: scaleX(1);
		}
	}
	@keyframes collapse {
		0% {
			transform: scaleX(1);
		}
		100% {
			transform: scaleX(0);
		}
	}

	:global(.step) {
		height: 100%;
		display: grid;
		grid-template-rows: var(--title-height) 1fr;
	}

	:global(.step > .title) {
		z-index: $COLUMN_TITLE_INDEX;
		display: flex;
		flex-direction: column;
		justify-content: end;
		grid-row: 1;
		color: theme('colors.gray.400');
		white-space: nowrap;
		padding-bottom: 2rem;
		overflow: visible;
		min-width: 0;
		transition: all 0.5s;
		cursor: default;

		&:hover {
			color: theme('colors.gray.600');
		}
	}

	:global(.step > .title.expandable) {
		cursor: pointer;
	}

	:global(.step .content) {
		grid-row: 2;
		height: fit-content;
	}

	:global(.column) {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		position: relative;

		:global(.cell) {
			height: var(--vector-height);
			display: flex;
			gap: 1rem;
			align-items: center;
			position: relative;
		}

		:global(.subtitle) {
			position: absolute;
			top: 0;
			transform: translateY(calc(-100% - 1rem));
			text-align: center;
			font-size: 0.8rem;
			color: theme('colors.gray.400');
			width: 100%;
			z-index: $COLUMN_TITLE_INDEX;
		}
	}

	:global(.vector),
	:global(.sub-vector) {
		position: relative;
		z-index: $VECTOR_INDEX;
		width: 12px;
		height: var(--vector-height);
		flex-shrink: 0;
		justify-content: start;
	}
	:global(.cell.x1-12),
	:global(.vector.x1-12),
	:global(.sub-vector.x1-12) {
		height: calc(var(--vector-height) / 12);
	}

	:global(.cell.x3),
	:global(.vector.x3),
	:global(.sub-vector.x3) {
		height: calc(var(--vector-height) * 3);
	}
	:global(.cell.x4),
	:global(.vector.x4),
	:global(.sub-vector.x4) {
		height: calc(var(--vector-height) * 3.1);
	}

	:global(.vector.vocab),
	:global(.sub-vector.vocab) {
		height: 100%;
		width: 0;
	}

	:global(.sub-vector.head-rest) {
		flex: 1 0 0;
	}

	:global(.label) {
		font-size: 0.9rem;
		color: theme('colors.gray.700');
		z-index: $VECTOR_INDEX;
		display: inline;
		max-width: 7rem;
		overflow: hidden;
		text-overflow: ellipsis;
		text-align: right;
		line-height: var(--vector-height);
		height: var(--vector-height);
		flex-shrink: 0;
	}
	:global(.label.float) {
		position: absolute;
		left: -0.8rem;
		transform: translateX(-100%);
	}
	:global(.label.float-right) {
		position: absolute;
		left: -0.8rem;
	}

	:global(.ellipsis) {
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	:global(.bounding) {
		position: absolute;
		box-sizing: content-box;
		top: -0.5rem;
		padding: 0.5rem 0;
		left: 0;
		height: 100%;
		border: 2px dashed theme('colors.gray.300');
		border-radius: 0.5rem;
		transition: opacity 0.5s;
		opacity: 0;
		pointer-events: none;
	}
	:global(.bounding.active) {
		opacity: 0.8;
	}

	:global(.popover) {
		z-index: $POPOVER_INDEX;
		width: max-content;
	}

	:global(.tooltip) {
		z-index: $TOOLTIP_INDEX;
		background-color: white !important;
		color: theme('colors.gray.600') !important;
		border: 1px solid theme('colors.gray.200') !important;
		padding: 0.2rem 0.5rem !important;
		font-size: 0.8rem !important;
		white-space: nowrap;
		font-weight: 300 !important;
		border-color: theme('colors.gray.200') !important;
	}
	.dim {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		z-index: $DIM_INDEX;
		background-color: white;
		opacity: 0.7;
		user-select: none;

		&.attention {
			z-index: 0;
		}
	}
	.dim-partial {
		user-select: none;
		z-index: $PARTIAL_DIM_INDEX;
		position: absolute;
		top: 0;
		height: 100%;

		&.right {
			right: 0;
			background: linear-gradient(90deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 1) 80%);
		}
		&.left {
			left: 0;
			background: linear-gradient(-90deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 1) 80%);
		}

		&.embedding {
			&.left {
				display: none;
			}
			&.right {
				width: 60%;
			}
		}
		&.attention {
			&.left {
				width: 20%;
			}
			&.right {
				width: 20%;
			}
		}
		&.softmax {
			&.left {
				width: 60%;
			}
			&.right {
				display: none;
			}
		}
	}
	.sankey {
		position: absolute;
		left: 0;
		top: 0;
		width: 100%;
		height: 100%;

		&.attention {
			:global(.sankey-top) {
				z-index: $EXPANDED_ATTENTION_INDEX !important;
				pointer-events: none;
			}
			// :global(.sankey-top > g) {
			// 	opacity: 0.3;
			// }
			// :global(.sankey-top > g.attention) {
			// 	opacity: 1;
			// }
		}
	}

	:global(svg g.path-group) {
		transition: opacity 0.5s;
	}
	:global(div.step > div) {
		transition: opacity 0.5s;
	}
	:global(div.step .column) {
		transition: opacity 0.5s;
	}
</style>
