import { drawPitchOutline } from './render_pitch.js';
import { resizeDrawable } from './drawable.js';
import { resizeTactical  } from './video_analysis.js';

const drawable = document.getElementById('drawable');
const tacticalboard = document.getElementById('tacticalboard');

const checkHtmlVideoElementCapabilities = () => {
    if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) {
        document.getElementById('supported-browsers-disclaimer').classList.remove('nodisplay');
    }

}

//---------------------------------------------
// EVENT LISTENERS
//---------------------------------------------

window.onload = () => {
    checkHtmlVideoElementCapabilities();
    resizeTactical();
    drawPitchOutline(tacticalboard);
    resizeDrawable();
}

window.addEventListener('resize', e => {
    resizeTactical();
    drawPitchOutline(tacticalboard);
    resizeDrawable();
    resizeDrawable();
});