// Canvas and context
const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');

let layers = [];
let layerCounter = 0;
let selectedLayer = null;

let viewOffsetX = 0, viewOffsetY = 0, viewScale = 1;
let isPanning = false, panStart = { x: 0, y: 0 };

// For dragging a selected layer individually
let draggingLayer = false;
let layerDragStart = { x: 0, y: 0 };
let initialLayerPos = { x: 0, y: 0 };

// Active control for resizing/rotating (handles)
let activeControl = null;

// Set up canvas size based on container dimensions
function resizeCanvas() {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  redrawCanvas();
  updateResizeOverlay();
}
window.addEventListener('resize', resizeCanvas);

// Show/hide bottom panels
function showBottomPanel(panelId) {
  document.getElementById('assets-panel').classList.add('hidden');
  document.getElementById('adjustments-panel').classList.add('hidden');
  const bottomPanel = document.getElementById('bottom-panel');
  if(panelId === 'assets' || panelId === 'adjustments'){
    document.getElementById(panelId + '-panel').classList.remove('hidden');
    bottomPanel.style.maxHeight = '400px';
  } else {
    bottomPanel.style.maxHeight = '0';
  }
}

// --- Layer dragging vs. canvas panning ---
canvas.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mouseX = (e.clientX - rect.left - viewOffsetX) / viewScale;
  const mouseY = (e.clientY - rect.top - viewOffsetY) / viewScale;
  // If a layer is selected and the click is inside its bounding box, start dragging the layer.
  if(selectedLayer &&
     mouseX >= selectedLayer.x - (selectedLayer.image.width * selectedLayer.scale) / 2 &&
     mouseX <= selectedLayer.x + (selectedLayer.image.width * selectedLayer.scale) / 2 &&
     mouseY >= selectedLayer.y - (selectedLayer.image.height * selectedLayer.scale) / 2 &&
     mouseY <= selectedLayer.y + (selectedLayer.image.height * selectedLayer.scale) / 2) {
    draggingLayer = true;
    layerDragStart.x = e.clientX;
    layerDragStart.y = e.clientY;
    initialLayerPos.x = selectedLayer.x;
    initialLayerPos.y = selectedLayer.y;
    return;
  }
  // Otherwise, start panning the canvas and change cursor to indicate panning.
  isPanning = true;
  panStart.x = e.clientX;
  panStart.y = e.clientY;
  canvas.style.cursor = 'grabbing';
});

canvas.addEventListener('mousemove', (e) => {
  if (draggingLayer && selectedLayer) {
    const dx = (e.clientX - layerDragStart.x) / viewScale;
    const dy = (e.clientY - layerDragStart.y) / viewScale;
    selectedLayer.x = initialLayerPos.x + dx;
    selectedLayer.y = initialLayerPos.y + dy;
    redrawCanvas();
    updateResizeOverlay();
  } else if (isPanning) {
    const dx = e.clientX - panStart.x;
    const dy = e.clientY - panStart.y;
    viewOffsetX += dx;
    viewOffsetY += dy;
    panStart.x = e.clientX;
    panStart.y = e.clientY;
    redrawCanvas();
    updateResizeOverlay();
  }
});

canvas.addEventListener('mouseup', () => { 
  isPanning = false; 
  draggingLayer = false;
  canvas.style.cursor = 'grab';
});
canvas.addEventListener('mouseleave', () => { 
  isPanning = false; 
  draggingLayer = false;
  canvas.style.cursor = 'grab';
});

// Click on canvas selects a layer (if within bounds)
canvas.addEventListener('click', (e) => {
  if(activeControl) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left - viewOffsetX) / viewScale;
  const y = (e.clientY - rect.top - viewOffsetY) / viewScale;
  selectedLayer = null;
  for (let i = layers.length - 1; i >= 0; i--) {
    const layer = layers[i];
    const halfW = (layer.image.width * layer.scale) / 2;
    const halfH = (layer.image.height * layer.scale) / 2;
    if (x >= layer.x - halfW && x <= layer.x + halfW &&
        y >= layer.y - halfH && y <= layer.y + halfH) {
      selectedLayer = layer;
      break;
    }
  }
  updateResizeOverlay();
});

// Redraw canvas
function redrawCanvas() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(viewScale, 0, 0, viewScale, viewOffsetX, viewOffsetY);
  layers.forEach(layer => {
    if(layer.visible) {
      ctx.save();
      ctx.translate(layer.x, layer.y);
      ctx.rotate(layer.rotation);
      ctx.scale(layer.scale, layer.scale);
      ctx.drawImage(layer.image, -layer.image.width / 2, -layer.image.height / 2);
      ctx.restore();
    }
  });
}

// Utility: get layer center on screen
function getLayerCenterOnScreen(layer) {
  const x = viewScale * layer.x + viewOffsetX;
  const y = viewScale * layer.y + viewOffsetY;
  return { x, y };
}

// Drag and drop events for canvas
canvas.addEventListener('dragover', (e) => {
  e.preventDefault();
  canvas.style.border = '2px dashed #10b981';
});
canvas.addEventListener('dragleave', () => { canvas.style.border = 'none'; });
canvas.addEventListener('drop', (e) => {
  e.preventDefault();
  canvas.style.border = 'none';
  if (e.dataTransfer.files.length > 0) {
    const file = e.dataTransfer.files[0];
    if (file) handleImageUpload(file);
  } else {
    const imgSrc = e.dataTransfer.getData('text/plain');
    if (imgSrc) {
      const img = new Image();
      img.onload = () => { addNewLayer(img); redrawCanvas(); };
      img.src = imgSrc;
    }
  }
});

// Handle image upload
function handleImageUpload(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => { 
      addNewLayer(img); 
      redrawCanvas(); 
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// Add a new layer with scaling and offset (to avoid "gluing")
function addNewLayer(img) {
  layerCounter++;
  let scale = 1;
  const maxWidth = canvas.width * 0.8;
  const maxHeight = canvas.height * 0.8;
  const widthRatio = maxWidth / img.width;
  const heightRatio = maxHeight / img.height;
  scale = Math.min(1, widthRatio, heightRatio);
  let centerX = (canvas.width / 2 - viewOffsetX) / viewScale;
  let centerY = (canvas.height / 2 - viewOffsetY) / viewScale;
  if (layers.length > 0) {
    const lastLayer = layers[layers.length - 1];
    centerX = lastLayer.x + 40;
    centerY = lastLayer.y + 40;
  }
  const layer = {
    id: layerCounter,
    image: img,
    visible: true,
    name: `Layer ${layerCounter}`,
    x: centerX,
    y: centerY,
    scale: scale,
    rotation: 0
  };
  layers.push(layer);
  updateLayersList();
}

function removeBackground() { 
  // Now handled via backend call in handleRemoveBg()
}

function downloadCanvas() {
  const link = document.createElement('a');
  link.download = 'design.png';
  link.href = canvas.toDataURL();
  link.click();
}

// Asset upload and creation
function handleAssetUpload(event) {
  const files = event.target.files;
  const assetsGrid = document.getElementById('assets-grid');
  for (const file of files) {
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const assetElement = createAssetElement(e.target.result, file.name);
        assetsGrid.insertBefore(assetElement, assetsGrid.firstChild);
      };
      reader.readAsDataURL(file);
    }
  }
}
function createAssetElement(src, name) {
  const div = document.createElement('div');
  div.className = 'group relative bg-neutral-800/50 rounded-xl p-2 hover:bg-neutral-800 transition-colors cursor-move';
  div.draggable = true;
  div.innerHTML = `
    <img src="${src}" alt="${name}" class="w-full h-24 object-cover rounded-lg mb-2">
    <p class="text-xs text-neutral-400 truncate">${name}</p>
    <button onclick="deleteAsset(this)" class="absolute top-1 right-1 p-1 bg-black/50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity">
      <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
      </svg>
    </button>
  `;
  div.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/plain', src); });
  return div;
}
function deleteAsset(button) { 
  button.closest('div').remove(); 
}

// Update layers list in the right panel
function updateLayersList() {
  const layersList = document.getElementById('layers-list');
  layersList.innerHTML = '';
  // Iterate in reverse order so that top layers are listed first.
  [...layers].reverse().forEach(layer => {
    // Determine the toggle icon based on visibility:
    const toggleIcon = layer.visible
      ? `<svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
         </svg>`
      : `<svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.542-7a10.05 10.05 0 012.143-3.357M4.707 4.707l14.586 14.586" />
           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.88 9.88a3 3 0 104.24 4.24" />
         </svg>`;
    // Create the layer element; clicking the element selects the layer.
    const layerElement = document.createElement('div');
    layerElement.className = 'bg-neutral-800/50 backdrop-blur-sm rounded-xl p-3 flex items-center justify-between group hover:bg-neutral-800 transition-colors cursor-pointer';
    layerElement.setAttribute('draggable', true);
    layerElement.setAttribute('onclick', `selectLayer(${layer.id})`);
    layerElement.innerHTML = `
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-lg bg-neutral-700 flex items-center justify-center overflow-hidden">
          <img src="${layer.image.src}" class="w-full h-full object-cover">
        </div>
        <span class="text-sm text-neutral-200">${layer.name}</span>
      </div>
      <div class="flex gap-2">
        <button onclick="toggleLayerVisibility(${layer.id}); event.stopPropagation();" class="text-neutral-400 hover:text-white" title="Toggle Visibility">
          ${toggleIcon}
        </button>
        <button onclick="deleteLayer(${layer.id}); event.stopPropagation();" class="text-neutral-400 hover:text-white" title="Delete Layer">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>
    `;
    // Drag and drop events for reordering layers:
    layerElement.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', layer.id);
    });
    layerElement.addEventListener('dragover', (e) => { e.preventDefault(); });
    layerElement.addEventListener('drop', (e) => {
      e.preventDefault();
      const draggedId = parseInt(e.dataTransfer.getData('text/plain'));
      reorderLayers(draggedId, layer.id);
    });
    layersList.appendChild(layerElement);
  });
}
function selectLayer(layerId) {
  selectedLayer = layers.find(l => l.id === layerId);
  updateResizeOverlay();
}
function toggleLayerVisibility(layerId) {
  const layer = layers.find(l => l.id === layerId);
  if(layer){
    layer.visible = !layer.visible;
    redrawCanvas();
    updateLayersList();
    updateResizeOverlay();
  }
}
function deleteLayer(layerId) {
  layers = layers.filter(l => l.id !== layerId);
  if(selectedLayer && selectedLayer.id === layerId){
    selectedLayer = null;
  }
  redrawCanvas();
  updateLayersList();
  updateResizeOverlay();
}
function reorderLayers(draggedId, targetId) {
  const draggedIndex = layers.findIndex(l => l.id === draggedId);
  const targetIndex = layers.findIndex(l => l.id === targetId);
  if(draggedIndex > -1 && targetIndex > -1){
    const [draggedLayer] = layers.splice(draggedIndex, 1);
    layers.splice(targetIndex, 0, draggedLayer);
    redrawCanvas();
    updateLayersList();
  }
}

// Update the resizing overlay position and size
function updateResizeOverlay() {
  const overlay = document.getElementById('resize-overlay');
  if(!selectedLayer) { 
    overlay.classList.add('hidden');
    return; 
  }
  const centerX = viewScale * selectedLayer.x + viewOffsetX;
  const centerY = viewScale * selectedLayer.y + viewOffsetY;
  const width = viewScale * selectedLayer.image.width * selectedLayer.scale;
  const height = viewScale * selectedLayer.image.height * selectedLayer.scale;
  overlay.style.left = (centerX - width/2) + 'px';
  overlay.style.top = (centerY - height/2) + 'px';
  overlay.style.width = width + 'px';
  overlay.style.height = height + 'px';
  overlay.classList.remove('hidden');
}

// Handle mousedown on a resize or rotate handle
function onHandleMouseDown(e) {
  e.stopPropagation();
  const handleType = e.target.getAttribute('data-handle');
  if(!selectedLayer) return;
  if(handleType === 'rotate') {
    activeControl = {
      type: 'rotate',
      initialMouse: { x: e.clientX, y: e.clientY },
      initialRotation: selectedLayer.rotation,
      center: getLayerCenterOnScreen(selectedLayer)
    };
  } else {
    activeControl = {
      type: 'resize',
      handle: handleType,
      initialMouse: { x: e.clientX, y: e.clientY },
      initialScale: selectedLayer.scale,
      center: getLayerCenterOnScreen(selectedLayer)
    };
    const dx = e.clientX - activeControl.center.x;
    const dy = e.clientY - activeControl.center.y;
    activeControl.initialDistance = Math.sqrt(dx*dx + dy*dy);
  }
}
// Attach mousedown events to all handles
const handles = document.querySelectorAll('#resize-overlay [data-handle]');
handles.forEach(handle => {
  handle.addEventListener('mousedown', onHandleMouseDown);
});
// Global mousemove for active control (resize/rotate)
window.addEventListener('mousemove', (e) => {
  if(activeControl && selectedLayer) {
    if(activeControl.type === 'rotate') {
      const center = activeControl.center;
      const dx = e.clientX - center.x;
      const dy = e.clientY - center.y;
      const newAngle = Math.atan2(dy, dx);
      const initialDx = activeControl.initialMouse.x - center.x;
      const initialDy = activeControl.initialMouse.y - center.y;
      const initialAngle = Math.atan2(initialDy, initialDx);
      const delta = newAngle - initialAngle;
      selectedLayer.rotation = activeControl.initialRotation + delta;
    } else if(activeControl.type === 'resize') {
      const center = activeControl.center;
      const dx = e.clientX - center.x;
      const dy = e.clientY - center.y;
      const currentDistance = Math.sqrt(dx*dx + dy*dy);
      const factor = currentDistance / activeControl.initialDistance;
      selectedLayer.scale = activeControl.initialScale * factor;
    }
    redrawCanvas();
    updateResizeOverlay();
  }
});
window.addEventListener('mouseup', () => { activeControl = null; });

// Auto-resizing textarea
document.querySelector('textarea').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = this.scrollHeight + 'px';
});

// ---------------- Backend integration functions ----------------

// When Remove Background is pressed, send canvas image to backend
async function handleRemoveBg() {
  const dataURL = canvas.toDataURL();
  const base64Image = dataURL.split(',')[1];
  const response = await fetch('/remove-bg', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ image: base64Image })
  });
  const result = await response.json();
  if (result.image) {
     const img = new Image();
     img.onload = () => { addNewLayer(img); redrawCanvas(); addAsset(img, "Removed BG"); };
     img.src = "data:image/png;base64," + result.image;
  } else {
     alert("Background removal failed");
  }
}

// When prompt arrow is pressed, send prompt to backend
async function handleGenerate() {
  const promptText = document.querySelector('textarea').value;
  if (!promptText.trim()) {
    alert("Please enter a prompt.");
    return;
  }
  const response = await fetch('/generate', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ prompt: promptText })
  });
  const result = await response.json();
  if (result.image) {
     const img = new Image();
     img.onload = () => { addNewLayer(img); redrawCanvas(); addAsset(img, "Generated Image"); };
     img.src = "data:image/png;base64," + result.image;
  } else {
     alert("Image generation failed");
  }
}

// Helper to add an asset to the assets grid
function addAsset(img, name) {
  const assetsGrid = document.getElementById('assets-grid');
  const assetElement = createAssetElement(img.src, name || "Asset");
  assetsGrid.insertBefore(assetElement, assetsGrid.firstChild);
}

// ---------------- Navigation actions ----------------
function handleNavAction(action) {
  switch(action) {
    case 'upload': /* handle upload */; break;
    case 'layers': showBottomPanel(''); break;
    case 'assets': showBottomPanel('assets'); break;
    case 'adjustments': showBottomPanel('adjustments'); break;
    case 'removeBg': handleRemoveBg(); break;
    case 'download': downloadCanvas(); break;
  }
}

// Initialize on window load
window.onload = () => {
  updateLayersList();
  resizeCanvas();
  // Attach click event for the prompt arrow
  document.getElementById('prompt-arrow').addEventListener('click', handleGenerate);
};