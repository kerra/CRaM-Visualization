<script>
	import { page } from '$app/stores';
	import InputForm from '~/components/InputForm.svelte';

	export let isActive;

	// Check if current page is about page
	$: isAboutPage = $page.url.pathname === '/about';
</script>

<div class="top-bar flex w-full items-center gap-4 px-10 py-2 pb-3" class:active={isActive}>
	<div class="logo text-bold text-gray-700" data-click="logo">
		Q<span class="small">WEN</span> + C<span class="small">R</span>AM E<span class="small">XPLAINER</span>
	</div>
	<div class="inputs flex grow items-center">
		<div class="input-wrapper w-full" class:active={isActive}>
			{#if !isAboutPage}
				<InputForm />
			{/if}
		</div>
	</div>
</div>

<style lang="scss">
	.top-bar {
		background: linear-gradient(to bottom, rgba(255, 255, 255, 1) 60%, rgba(255, 255, 255, 0) 100%);
		.input-wrapper {
			&.active {
				opacity: 1;
				pointer-events: initial;
			}
			opacity: 0;
			transition: opacity 0.2s;
			pointer-events: none;
		}
		.logo {
			flex-shrink: 0;
			white-space: nowrap;
			font-family: 'Jersey 10', sans-serif;

			font-optical-sizing: auto;
			font-style: normal;

			font-size: 2rem;
			// color: theme('colors.blue.800');

			background: linear-gradient(
				to right,
				theme('colors.blue.500') 0%,
				theme('colors.purple.500') 60%
			);
			-webkit-background-clip: text;
			-webkit-text-fill-color: transparent;

			.small {
				font-size: 1.8rem;
			}
		}
	}
	@media (max-width: 900px) {
		.top-bar {
			gap: 0.5rem;
			padding-left: 0.75rem;
			padding-right: 0.75rem;
		}
		.top-bar .logo {
			font-size: 1.45rem;
		}
		.top-bar .logo .small {
			font-size: 1.25rem;
		}
		:global(.parameters) {
			display: none !important;
		}
	}
</style>
