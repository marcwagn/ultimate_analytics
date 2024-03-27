// wait for the content of the window element 
// to load, then performs the operations. 
// This is considered best practice. 
window.addEventListener('load', ()=>{ 
		
	resize(); // Resizes the drawable once the window loads 
	document.addEventListener('mousedown', startPainting); 
	document.addEventListener('mouseup', stopPainting); 
	document.addEventListener('mousemove', sketch); 
	//window.addEventListener('resize', resize); 
}); 
	
const drawable = document.querySelector('#drawable'); 
// Context for the drawable for 2 dimensional operations 
const drawable_ctx = drawable.getContext('2d'); 
const toolbar = document.getElementById('toolbar');



// Resizes the drawable to the available size of the window. 
function resize(){ 
    let style = getComputedStyle(drawable);
    drawable_ctx.canvas.width = parseInt(style.width);
    drawable_ctx.canvas.height = parseInt(style.height); 
} 
	
// Stores the initial position of the cursor 
let coord = {x:0 , y:0}; 

// This is the flag that we are going to use to 
// trigger drawing 
let paint = false; 
	
// Updates the coordianates of the cursor when 
// an event e is triggered to the coordinates where 
// the said event is triggered. 
function getPosition(event){ 
    coord.x = event.clientX - drawable.offsetLeft; 
    coord.y = event.clientY - drawable.offsetTop; 
} 

// The following functions toggle the flag to start 
// and stop drawing 
function startPainting(event){ 
    paint = true; 
    getPosition(event); 
} 
function stopPainting(){ 
    paint = false; 
} 
	
function sketch(event){ 
if (!paint) return; 
drawable_ctx.beginPath(); 

drawable_ctx.lineWidth = 5; 

// Sets the end of the lines drawn 
// to a round shape. 
drawable_ctx.lineCap = 'round'; 
	
drawable_ctx.strokeStyle = 'green'; 
	
// The cursor to start drawing 
// moves to this coordinate 
drawable_ctx.moveTo(coord.x, coord.y); 

// The position of the cursor 
// gets updated as we move the 
// mouse around. 
getPosition(event); 

// A line is traced from start 
// coordinate to this coordinate 
drawable_ctx.lineTo(coord.x , coord.y); 
	
// Draws the line. 
drawable_ctx.stroke(); 
}
