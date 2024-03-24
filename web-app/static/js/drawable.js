const canvas = document.getElementById('drawable');
const toolbar = document.getElementById('toolbar');
const ctx = canvas.getContext('2d');

canvasOffsetX = canvas.offsetLeft;
canvasOffsetY = canvas.offsetTop;

let style = getComputedStyle(drawable);
ctx.canvas.width = parseInt(style.width);
ctx.canvas.height = parseInt(style.height); 

let isPainting = false;
let lineWidth = 5;
let startX;
let startY;

window.addEventListener('resize', resize);

function resize(){ 
    let style = getComputedStyle(drawable);
    ctx.canvas.width = parseInt(style.width);
    ctx.canvas.height = parseInt(style.height);

    canvasOffsetX = canvas.offsetLeft;
    canvasOffsetY = canvas.offsetTop;
} 

toolbar.addEventListener('click', e => {
    if (e.target.id === 'clear') {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
});

toolbar.addEventListener('change', e => {
    if(e.target.id === 'stroke') {
        ctx.strokeStyle = e.target.value;
    }

    if(e.target.id === 'lineWidth') {
        lineWidth = e.target.value;
    }
    
});

const draw = (e) => {
    if(!isPainting) {
        return;
    }

    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';

    ctx.lineTo(e.clientX - canvasOffsetX, e.clientY - canvasOffsetY);
    ctx.stroke();
}

canvas.addEventListener('mousedown', (e) => {
    isPainting = true;
    startX = e.clientX;
    startY = e.clientY;
});

canvas.addEventListener('mouseup', e => {
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);
