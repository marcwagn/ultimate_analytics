import { drawPitchOutline } from './render_pitch.js';
import { resizeDrawable } from './drawable.js';

const drawable = document.getElementById('drawable');
const tacticalboard = document.getElementById('tacticalboard');
const controlsContainer = document.getElementById('controls-container');


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