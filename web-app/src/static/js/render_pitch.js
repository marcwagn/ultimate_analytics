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

const drawPitchOutline = () => {
    var tacticalboard = document.getElementById('tacticalboard');
    var ctx_tacticalboard = tacticalboard.getContext('2d');

    let vertical_pitch_offset = 10;
    let pitch_height = tacticalboard.height - 2*vertical_pitch_offset;
    let pitch_width = pitch_height * 0.37;
    let horizontal_pitch_offset = (tacticalboard.width - pitch_width) / 2;

    const drawLine = (ctx, x1, y1, x2, y2) => {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      };

    const drawCircle = (ctx, x, y, radius) => {
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'white';
        ctx.fill();
      };
    
    //render gras background and draw pitch outline
    var img = new Image();
    img.src = 'static/img/gras.jpg';

    img.addEventListener('load', () => {
        // render gras background
        const ptrn = ctx_tacticalboard.createPattern(img, 'repeat');
        ctx_tacticalboard.fillStyle = ptrn;
        ctx_tacticalboard.fillRect(0, 0, tacticalboard.width, tacticalboard.height);
        // draw rectangular outline
        ctx_tacticalboard.strokeStyle = 'white'; 
        ctx_tacticalboard.lineWidth = 5;
        ctx_tacticalboard.strokeRect(   horizontal_pitch_offset, vertical_pitch_offset, 
                                        pitch_width, pitch_height);
        // draw top endzone line
        drawLine(ctx_tacticalboard, 
            horizontal_pitch_offset, vertical_pitch_offset + pitch_height*0.18, 
            horizontal_pitch_offset + pitch_width, vertical_pitch_offset + pitch_height*0.18);
        // draw bottom endzone line
        drawLine(ctx_tacticalboard, 
            horizontal_pitch_offset, vertical_pitch_offset + pitch_height* (1-0.18), 
            horizontal_pitch_offset + pitch_width, vertical_pitch_offset + pitch_height*(1-0.18));
        
        // draw top break point
        drawCircle(ctx_tacticalboard, 
            horizontal_pitch_offset + pitch_width/2, vertical_pitch_offset + pitch_height*0.32, 5);
        // draw bottom break point
        drawCircle(ctx_tacticalboard,
            horizontal_pitch_offset + pitch_width/2, vertical_pitch_offset + pitch_height*(1-0.32), 5);
      });
}

//---------------------------------------------
// EVENT LISTENERS
//---------------------------------------------

window.onload = () => {
    resizeTactical();
    drawPitchOutline();
    resizeDrawable();
    document.getElementById('dashboard-container').classList.remove('hidden')
}

window.addEventListener('resize', e => {
    resizeTactical();
    drawPitchOutline();
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
