const $ = (selector) => document.querySelector(selector);

/**
 * Draws a line on the canvas context.
 *
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 * @param {number} x1 - The x-coordinate of the starting point.
 * @param {number} y1 - The y-coordinate of the starting point.
 * @param {number} x2 - The x-coordinate of the ending point.
 * @param {number} y2 - The y-coordinate of the ending point.
 */
const drawLine = (ctx, x1, y1, x2, y2) => {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
};

/**
 * Draws a circle on the canvas.
 *
 * @param {CanvasRenderingContext2D} ctx - The rendering context of the canvas.
 * @param {number} x - The x-coordinate of the center of the circle.
 * @param {number} y - The y-coordinate of the center of the circle.
 * @param {number} radius - The radius of the circle.
 * @param {string} color - The color of the circle.
 */
export const drawCircle = (ctx, x, y, radius, color) => {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
};

/**
 * Calculates the dimensions of the pitch based on the provided tactical board context.
 *
 * @param {Object} tacticalboard - The tactical board.
 * @returns {number} .vertical_pitch_offset - The vertical offset of the pitch.
 * @returns {number} .horizontal_pitch_offset - The horizontal offset of the pitch.
 * @returns {number} .pitch_height - The height of the pitch.
 * @returns {number} .pitch_width - The width of the pitch.
 */
export const getPitchDimensions = (tacticalboard)=> {
    const vertical_pitch_offset = 10;
    const pitch_height = tacticalboard.height - 2*vertical_pitch_offset;
    const pitch_width = pitch_height * 0.37;
    const horizontal_pitch_offset = (tacticalboard.width - pitch_width) / 2;

    return {vertical_pitch_offset, horizontal_pitch_offset, pitch_height, pitch_width};
}

/**
 * Converts standard coordinates to canvas coordinates based on the given tactical board.
 *
 * @param {number} x - The x-coordinate in standard coordinates.
 * @param {number} y - The y-coordinate in standard coordinates.
 * @param {object} tacticalboard - The tactical board object.
 * @returns {object} - The converted coordinates as an object with x and y properties.
 */
export const standardCoordsToCanvasCoords = (x, y, tacticalboard) => {
    const { vertical_pitch_offset, horizontal_pitch_offset, 
        pitch_height, pitch_width } = getPitchDimensions(tacticalboard);
    
    const canvasX = Math.round(x * (pitch_width / 37) + horizontal_pitch_offset);
    const canvasY = Math.round(y * (pitch_height / 100) + vertical_pitch_offset);

    return {x: canvasX, y: canvasY};
}

/**
 * Draws the pitch outline on the tactical board canvas.
 * 
 * @param {Object} tacticalboard - The tactical board.
 */
export const drawPitchOutline = (tacticalboard) => {
    const ctx_tacticalboard = tacticalboard.getContext('2d');

    // get dimensions of the pitch
    const { vertical_pitch_offset, horizontal_pitch_offset, 
            pitch_height, pitch_width } = getPitchDimensions(tacticalboard);

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
            horizontal_pitch_offset + pitch_width/2, vertical_pitch_offset + pitch_height*0.32, 3, "white");
        // draw bottom break point
        drawCircle(ctx_tacticalboard,
            horizontal_pitch_offset + pitch_width/2, vertical_pitch_offset + pitch_height*(1-0.32), 3, "white");
      });
}