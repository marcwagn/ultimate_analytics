import { drawPitchOutline } from './render_pitch.js';

const $ = (selector) => document.querySelector(selector);

// Load control elements
const controlsContainer = document.getElementById('controls-container');
const tacticalboard = document.getElementById('tacticalboard');
const drawable = document.getElementById('drawable');
const toolbar = document.getElementById('toolbar');

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

//---------------------------------------------
// FUNCTIONS
//---------------------------------------------

const resizeTactical = () => {
    const controlsContainerHeight = controlsContainer.offsetHeight;
  
    let offsetHeight = 25;
    let controlsHeight = controlsContainerHeight - 2*offsetHeight;
    let controlsWidth = controlsContainerHeight / 1.5;
  
    tacticalboard.width = controlsWidth;
    tacticalboard.height = controlsHeight;
  
    drawable.width = controlsWidth;
    drawable.height = controlsHeight;
  }

const resizeDrawable = () => { 
    style = getComputedStyle(drawable);
    drawable_ctx.canvas.width = parseInt(style.width);
    drawable_ctx.canvas.height = parseInt(style.height);

    canvasOffsetX = drawable.offsetLeft;
    canvasOffsetY = drawable.offsetTop;
} 

//---------------------------------------------
// EVENT LISTENERS
//---------------------------------------------

window.onload = () => {
    resizeTactical();
    drawPitchOutline(tacticalboard);
    resizeDrawable();
    document.getElementById('dashboard-container').classList.remove('hidden')
}

window.addEventListener('resize', e => {
    resizeTactical();
    drawPitchOutline(tacticalboard);
    resizeDrawable();
    resizeDrawable();
});

//---------------------------------------------
// Drawable Event Listeners
//---------------------------------------------
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
