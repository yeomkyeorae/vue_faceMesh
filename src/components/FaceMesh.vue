<template>
	<div class="illustrationDiv">
		<div id="loading" style='position: relative; left: 0'>
			<span class="spinner-text" id="status">
				Loading PoseNet model...
			</span>
			<div class="sk-spinner sk-spinner-pulse"></div>
		</div>
		<div class="canvas-container">
			<div id='main' style='display:none'>
				<video id="video" playsinline style=" -moz-transform: scaleX(-1);
            -o-transform: scaleX(-1);
            -webkit-transform: scaleX(-1);
            transform: scaleX(-1);
            display: none;
            ">
				</video>
				<canvas id="output" class="camera-canvas"></canvas>
				<canvas id="keypoints" class="camera-canvas"></canvas>
			</div>
			<canvas class="illustration-canvas" />
		</div>
		<!-- <script src="camera.js"></script> -->
	</div>
</template>

<script>
	export default {
		name: 'FaceMesh'
	}
</script>

<style>
	.illustrationDiv {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
	}

	.canvas-container {
		width: 800px;
		max-width: 100%;
		display: flex;
		justify-content: center;
		position: relative;
	}

	.camera-canvas {
		position: absolute;
		transform: scale(0.5, 0.5);
		transform-origin: 0 0;
		left: 10px;
		top: 10px;
	}

	#main {
		left: 0;
		top: 0;
		position: absolute;
	}

	.illustration-canvas {
		border: 1px solid #eeeeee;
	}

	.footer {
		position: fixed;
		left: 0;
		bottom: 0;
		width: 100%;
		color: black;
	}

	.footer-text {
		max-width: 600px;
		text-align: center;
		margin: auto;
	}

	@media only screen and (max-width: 600px) {

		.footer-text,
		.dg {
			display: none;
		}
	}

	/*
         *  The following loading spinner CSS is from SpinKit project
         *  https://github.com/tobiasahlin/SpinKit
         */
	.sk-spinner-pulse {
		width: 20px;
		height: 20px;
		margin: auto 10px;
		float: left;
		background-color: #333;
		border-radius: 100%;
		-webkit-animation: sk-pulseScaleOut 1s infinite ease-in-out;
		animation: sk-pulseScaleOut 1s infinite ease-in-out;
	}

	@-webkit-keyframes sk-pulseScaleOut {
		0% {
			-webkit-transform: scale(0);
			transform: scale(0);
		}

		100% {
			-webkit-transform: scale(1.0);
			transform: scale(1.0);
			opacity: 0;
		}
	}

	@keyframes sk-pulseScaleOut {
		0% {
			-webkit-transform: scale(0);
			transform: scale(0);
		}

		100% {
			-webkit-transform: scale(1.0);
			transform: scale(1.0);
			opacity: 0;
		}
	}

	.spinner-text {
		float: left;
	}
</style>