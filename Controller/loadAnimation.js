const animationVideo = document.createElement('video');
animationVideo.src = '/View/assets/SquatAnimation.mp4';
animationVideo.crossOrigin = 'anonymous'; // For CORS issues
animationVideo.loop = true;

animationVideo.addEventListener('loadeddata', () => {
    animationVideo.play(); // Start playing once the video is ready
    render();
});

const fbxCanvas = document.getElementById('fbxCanvas');
const context = fbxCanvas.getContext('2d');

function render() {
    context.clearRect(0, 0, fbxCanvas.width, fbxCanvas.height);
    context.drawImage(animationVideo, 0, 0, fbxCanvas.width, fbxCanvas.height);
    requestAnimationFrame(render);
}
