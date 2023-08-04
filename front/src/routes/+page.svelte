<script>
    let count = 0
    
    function incrementCounter() {
        count += 1
    }
</script>

<svelte:head>
	<title>Home</title>
	<meta name="description" content="Svelte demo app" />
</svelte:head>

<section class='flex justify-center items-center w-full h-screen gap-4 bg-slate-900 text-white'>
    <p class='font-medium text-lg'>Counter app: </p>
    <button
        class='rounded bg-indigo-500 px-4 py-2'
        on:click={incrementCounter}
    >
        {count}
    </button>
</section>

