// Load control elements
const toolbar = document.getElementById('toolbar');

const drawable = document.getElementById('drawable');
const drawable_ctx = drawable.getContext('2d');

// set variables for drawable canvas
let canvasOffsetX = drawable.offsetLeft;
let canvasOffsetY = drawable.offsetTop;

let style = getComputedStyle(drawable);
drawable_ctx.canvas.width = parseInt(style.width);
drawable_ctx.canvas.height = parseInt(style.height); 

let isPainting = false;
let lineWidth = 5;
let startX;
let startY;

export const resizeDrawable = () => { 
    style = getComputedStyle(drawable);
    drawable_ctx.canvas.width = parseInt(style.width);
    drawable_ctx.canvas.height = parseInt(style.height);

    canvasOffsetX = drawable.offsetLeft;
    canvasOffsetY = drawable.offsetTop;
} 

toolbar.addEventListener('click', e => {
    if (e.target.id === 'clear') {
        drawable_ctx.clearRect(0, 0, drawable.width, drawable.height);
    }
});

toolbar.addEventListener('change', e => {
    if(e.target.id === 'stroke') {
        drawable_ctx.strokeStyle = e.target.value;
    }

    if(e.target.id === 'lineWidth') {
        lineWidth = e.target.value;
    }
    
});

const draw = (e) => {
    if(!isPainting) {
        return;
    }

    drawable_ctx.lineWidth = lineWidth;
    drawable_ctx.lineCap = 'round';

    drawable_ctx.lineTo(e.clientX - canvasOffsetX, e.clientY - canvasOffsetY);
    drawable_ctx.stroke();
}

drawable.addEventListener('mousedown', (e) => {
    isPainting = true;
    startX = e.clientX;
    startY = e.clientY;
});

drawable.addEventListener('mouseup', e => {
    isPainting = false;
    drawable_ctx.stroke();
    drawable_ctx.beginPath();
});

drawable.addEventListener('mousemove', draw);
