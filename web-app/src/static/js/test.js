const controlsContainer = document.getElementById('controls-container');

export const resizeTactical = () => {
    const controlsContainerHeight = controlsContainer.offsetHeight;
  
    let offsetHeight = 25;
    let controlsHeight = controlsContainerHeight - 2*offsetHeight;
    let controlsWidth = controlsContainerHeight / 1.5;
  
    tacticalboard.width = controlsWidth;
    tacticalboard.height = controlsHeight;
  
    drawable.width = controlsWidth;
    drawable.height = controlsHeight;
  }