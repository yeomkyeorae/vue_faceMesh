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
			<canvas class="illustration-canvas"></canvas>
		</div>
		<!-- <script src="camera.js"></script> -->
	</div>
</template>

<script>
	import * as posenet_module from '@tensorflow-models/posenet';
	import * as facemesh_module from '@tensorflow-models/facemesh';
	import * as tf from '@tensorflow/tfjs'
	import * as paper from 'paper'
	import "babel-polyfill"

	import { drawKeypoints, drawPoint, drawSkeleton, toggleLoadingUI, setStatusText } from './utils/demoUtils';
	import { SVGUtils } from './utils/svgUtils'
	import { PoseIllustration } from './illustrationGen/illustration';
	import { Skeleton, facePartName2Index } from './illustrationGen/skeleton';
	// import { FileUtils } from './utils/fileUtils';

	import * as girlSVG from './resources/illustration/girl.svg';
	import * as boySVG from './resources/illustration/boy.svg';

	export default {
		name: 'FaceMesh',
		data() {
			return {
				// Camera stream video element
				video: {},
				videoWidth: 300,
				videoHeight: 300,

				// Canvas
				faceDetection: null,
				illustration: null,
				canvasScope: null,
				canvasWidth: 800,
				canvasHeight: 800,

				// ML models
				facemesh: null,
				posenet: null,
				minPoseConfidence: 0.15,
				minPartConfidence: 0.1,
				nmsRadius: 30.0,

				// Misc
				// mobile: false,
				// const stats = new Stats();
				avatarSvgs: {
					'girl': girlSVG.default,
					'boy': boySVG.default,
				},

				defaultPoseNetArchitecture: 'MobileNetV1',
				defaultQuantBytes: 2,
				defaultMultiplier: 1.0,
				defaultStride: 16,
				defaultInputResolution: 200,
			}
		},
		methods: {
			async setupCamera() {
				if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
					throw new Error(
						'Browser API navigator.mediaDevices.getUserMedia not available');
				}

				const video = document.getElementById('video');
				video.width = this.videoWidth;
				video.height = this.videoHeight;

				const stream = await navigator.mediaDevices.getUserMedia({
					'audio': false,
					'video': {
						facingMode: 'user',
						width: this.videoWidth,
						height: this.videoHeight,
					},
				});
				video.srcObject = stream;

				return new Promise((resolve) => {
					video.onloadedmetadata = () => {
						resolve(video);
					};
				});
			},
			async loadVideo() {
				const video = await this.setupCamera();
				video.play();

				return video;
			},
			detectPoseInRealTime(video) {
				const canvas = document.getElementById('output');
				const keypointCanvas = document.getElementById('keypoints');
				const videoCtx = canvas.getContext('2d');
				const keypointCtx = keypointCanvas.getContext('2d');

				canvas.width = this.videoWidth;
				canvas.height = this.videoHeight;
				keypointCanvas.width = this.videoWidth;
				keypointCanvas.height = this.videoHeight;

				let th = this;
				async function poseDetectionFrame() {
					let poses = [];

					videoCtx.clearRect(0, 0, th.videoWidth, th.videoHeight);
					// Draw video
					videoCtx.save();
					videoCtx.scale(-1, 1);
					videoCtx.translate(-th.videoWidth, 0);
					videoCtx.drawImage(video, 0, 0, th.videoWidth, th.videoHeight);
					videoCtx.restore();

					// Creates a tensor from an image
					const input = tf.browser.fromPixels(canvas);
					th.faceDetection = await th.facemesh.estimateFaces(input, false, false);
					let all_poses = await th.posenet.estimatePoses(video, {
						flipHorizontal: true,
						decodingMethod: 'multi-person',
						maxDetections: 1,
						scoreThreshold: th.minPartConfidence,
						nmsRadius: th.nmsRadius
					});

					poses = poses.concat(all_poses);
					input.dispose();

					keypointCtx.clearRect(0, 0, th.videoWidth, th.videoHeight);
					if (th.guiState.debug.showDetectionDebug) {
						poses.forEach(({
							score,
							keypoints
						}) => {
							if (score >= th.minPoseConfidence) {
								drawKeypoints(keypoints, th.minPartConfidence, keypointCtx);
								drawSkeleton(keypoints, th.minPartConfidence, keypointCtx);
							}
						});
						th.faceDetection.forEach(face => {
							Object.values(facePartName2Index).forEach(index => {
								let p = face.scaledMesh[index];
								drawPoint(keypointCtx, p[1], p[0], 2, 'red');
							});
						});
					}

					th.canvasScope.project.clear();

					if (poses.length >= 1 && th.illustration) {
						Skeleton.flipPose(poses[0]);

						if (th.faceDetection && th.faceDetection.length > 0) {
							let face = Skeleton.toFaceFrame(th.faceDetection[0]);
							th.illustration.updateSkeleton(poses[0], face);
						} else {
							th.illustration.updateSkeleton(poses[0], null);
						}
						th.illustration.draw(th.canvasScope, th.videoWidth, th.videoHeight);

						if (th.guiState.debug.showIllustrationDebug) {
							th.illustration.debugDraw(th.canvasScope);
						}
					}

					th.canvasScope.project.activeLayer.scale(
						th.canvasWidth / th.videoWidth,
						th.canvasHeight / th.videoHeight,
						new th.canvasScope.Point(0, 0));

					// End monitoring code for frames per second
					// stats.end();

					requestAnimationFrame(poseDetectionFrame);
				}

				poseDetectionFrame();
			},
			setupCanvas() {
				// this.mobile = isMobile();
				// if (this.mobile) {
				// 	this.canvasWidth = Math.min(window.innerWidth, window.innerHeight);
				// 	this.canvasHeight = this.canvasWidth;
				// 	this.videoWidth *= 0.7;
				// 	this.videoHeight *= 0.7;
				// }

				this.canvasScope = paper.default;
				let canvas = document.querySelector('.illustration-canvas');
				canvas.width = this.canvasWidth;
				canvas.height = this.canvasHeight;
				this.canvasScope.setup(canvas);
			},
			async bindPage() {
				this.setupCanvas();

				toggleLoadingUI(true);
				setStatusText('Loading PoseNet model...');
				this.posenet = await posenet_module.load({
					architecture: this.defaultPoseNetArchitecture,
					outputStride: this.defaultStride,
					inputResolution: this.defaultInputResolution,
					multiplier: this.defaultMultiplier,
					quantBytes: this.defaultQuantBytes
				});
				setStatusText('Loading FaceMesh model...');
				this.facemesh = await facemesh_module.load();

				setStatusText('Loading Avatar file...');
				await this.parseSVG(Object.values(this.avatarSvgs)[0]);

				setStatusText('Setting up camera...');
				try {
					this.video = await this.loadVideo();
				} catch (e) {
					let info = document.getElementById('info');
					info.textContent = 'this device type is not supported yet, ' +
						'or this browser does not support video capture: ' + e.toString();
					info.style.display = 'block';
					throw e;
				}

				// setupGui([], posenet);
				// setupFPS();

				toggleLoadingUI(false);
				this.detectPoseInRealTime(this.video, this.posenet);
			},
			async parseSVG(target) {
				let svgScope = await SVGUtils.importSVG(target /* SVG string or file path */ );
				let skeleton = new Skeleton(svgScope);
				this.illustration = new PoseIllustration(this.canvasScope);
				this.illustration.bindSkeleton(skeleton, svgScope);
			}
		},
		mounted() {
			this.guiState = {
				avatarSVG: Object.keys(this.avatarSvgs)[0],
				debug: {
					showDetectionDebug: true,
					showIllustrationDebug: false,
				},
			}
			navigator.getUserMedia = navigator.getUserMedia ||
				navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
			// FileUtils.setDragDropHandler((result) => {
			// 	this.parseSVG(result)
			// });
			this.bindPage();
		}
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