<script lang="ts">
	import '~/styles/app.css';
	import '~/styles/global.scss';
	import Topbar from '~/components/Topbar.svelte';
	import { isLoaded, predictedColor, rootRem, userId } from '~/store';
	import { onMount } from 'svelte';
	import { Spinner } from 'flowbite-svelte';
	import { page } from '$app/stores';

	let topBarHeight = 0;
	let scrollLeft = 0;

	let minScreenWidth = 1300;
	let minColumWidth = Math.floor(minScreenWidth / 24) - rootRem * 2;

	let intersectionObserver: IntersectionObserver;
	let tobBarActive = false;
	let target: HTMLElement;

	onMount(() => {
		isLoaded.set(true);

		const userIdParam = $page.url.searchParams.get('userId');

		if (userIdParam) {
			userId.set(userIdParam);
			(window as any).dataLayer?.push({
				event: 'user_identified',
				user_id: userIdParam,
				timestamp: new Date().toISOString()
			});
		}

		intersectionObserver = new IntersectionObserver(handleIntersection, {
			root: null,
			rootMargin: '0px',
			threshold: 0.2
		});

		if (target) {
			intersectionObserver.observe(target);
		}
		window.addEventListener('scroll', handleMobileScrollX);

		return () => {
			intersectionObserver.disconnect();
			window.removeEventListener('scroll', handleMobileScrollX);
		};
	});

	function handleIntersection(entries: any[]) {
		entries.forEach((entry: any) => {
			if (entry.isIntersecting) {
				tobBarActive = true;
			} else {
				tobBarActive = false;
			}
		});
	}

	const handleMobileScrollX = () => {
		scrollLeft = window.scrollX || document.documentElement.scrollLeft;
	};
</script>

<div
	id="app"
	style={`--min-screen-width:${minScreenWidth}px;--min-column-width:${minColumWidth}px;--predicted-color:${predictedColor};`}
>
	<div id="landing">
		<header bind:offsetHeight={topBarHeight} style="transform: translateX({-1 * scrollLeft}px);">
			<Topbar isActive={tobBarActive} />
		</header>
		<main id="main" style={`padding-top:${topBarHeight}px`} bind:this={target}>
			{#if $isLoaded}
				<slot />
			{:else}
				<div class="flex h-full w-full items-center justify-center"><Spinner color="purple" /></div>
			{/if}
		</main>
	</div>
</div>

<style lang="scss">
	#app {
		height: 100vh;
		width: 100vw;
		overflow: hidden;
	}

	#landing {
		height: 100%;
		width: 100%;
	}

	header {
		width: 100%;
		position: fixed;
		z-index: $TOP_BAR_INDEX;
	}
	main {
		position: relative;
		height: 95%;
		width: 100%;
		display: flex;
		justify-content: start;
		overflow: hidden;
	}
</style>
