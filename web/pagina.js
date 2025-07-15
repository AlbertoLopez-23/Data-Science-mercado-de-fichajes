let playersData = [];
let filteredData = [];
let csvHeaders = [];
let allPlayers = [];
let currentResults = [];
let currentSortColumn = null;
let currentSortDirection = 'asc';
let currentTableLimit = 500;
let filteredTableData = [];

// Configuración de columnas para la tabla (simplificada según solicitud)
const tableColumns = [
    { key: 'Nombre completo', label: 'Nombre', sortable: true },
    { key: 'Nacionalidad', label: 'Nacionalidad', sortable: true },
    { key: 'Posición principal', label: 'Posición', sortable: true },
    { key: 'Club actual', label: 'Club', sortable: true },
    { key: 'Fin de contrato', label: 'Fin Contrato', sortable: true },
    { key: 'Valor de mercado actual (numérico)', label: 'Valor Actual', sortable: true, type: 'currency' },
    { key: 'Valor_Predicho', label: 'Valor Predicho', sortable: true, type: 'predicted' },
    { key: 'Diferencia_Valor', label: 'Diferencia Valor', sortable: true, type: 'difference_colored' },
    { key: 'diferencia_relativa', label: 'Diferencia (%)', sortable: true, type: 'percentage' }
];

// Variables para filtros
let priceFilters = {
    minCurrent: 0,
    maxCurrent: 200000000,
    minPredicted: 0,
    maxPredicted: 200000000
};
let positionFilters = new Set();

// Referencias DOM
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const resultsGrid = document.getElementById('results-grid');
const resultsCount = document.getElementById('results-count');
const sortSelect = document.getElementById('sort-select');
const playerModal = document.getElementById('player-modal');
const modalClose = document.getElementById('modal-close');
const searchSection = document.getElementById('search-section');
const resultsSection = document.getElementById('results-section');
const searchError = document.getElementById('search-error');
const searchLoading = document.getElementById('search-loading');
const dataLoading = document.getElementById('data-loading');
const tableSection = document.getElementById('table-section');
const tableBtn = document.getElementById('table-btn');
const backToSearchBtn = document.getElementById('back-to-search-btn');
const limitSelect = document.getElementById('limit-select');
const tableFilter = document.getElementById('table-filter');
const playersTable = document.getElementById('players-table');
const tableHeaders = document.getElementById('table-headers');
const tableBody = document.getElementById('table-body');
const tableInfo = document.getElementById('table-info');

// Cargar datos automáticamente al iniciar la página
// document.addEventListener('DOMContentLoaded', loadData); // Removido - duplicado

// Event Listeners
searchBtn.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});
sortSelect.addEventListener('change', sortResults);
modalClose.addEventListener('click', closeModal);
playerModal.addEventListener('click', (e) => {
    if (e.target === playerModal) closeModal();
});

// Quick search buttons
document.querySelectorAll('.quick-search-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        searchInput.value = e.target.dataset.query;
        performSearch();
    });
});

// Función para cargar datos automáticamente
async function loadData() {
    const dataLoading = document.getElementById('data-loading');
    const uploadSuccess = document.getElementById('upload-success');
    const uploadError = document.getElementById('upload-error');
    
    try {
        dataLoading.style.display = 'block';
        uploadSuccess.style.display = 'none';
        uploadError.style.display = 'none';
        
        const response = await fetch('data.csv');
        if (!response.ok) {
            throw new Error(`Error al cargar el archivo: ${response.status} ${response.statusText}`);
        }
        
        const csvText = await response.text();
        
        Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            complete: function(results) {
                if (results.errors.length > 0) {
                    console.warn('Errores en el CSV:', results.errors);
                }
                
                allPlayers = results.data.filter(player => 
                    player['Nombre completo'] && 
                    player['Nombre completo'].trim() !== ''
                );
                
                // Mantener compatibilidad con el código existente
                filteredData = [...allPlayers];
                csvHeaders = results.meta.fields || [];
                
                dataLoading.style.display = 'none';
                uploadSuccess.innerHTML = `✅ Datos cargados exitosamente: ${allPlayers.length} jugadores`;
                uploadSuccess.style.display = 'block';
                searchSection.style.display = 'block';
                
                console.log('Datos cargados:', allPlayers.length, 'jugadores');
                console.log('Ejemplo de jugador:', allPlayers[0]);
            },
            error: function(error) {
                throw error;
            }
        });
        
    } catch (error) {
        console.error('Error al cargar datos:', error);
        dataLoading.style.display = 'none';
        uploadError.innerHTML = `❌ Error al cargar el archivo data.csv: ${error.message}`;
        uploadError.style.display = 'block';
    }
}

function updateSortOptions() {
    const commonSortFields = {
        'Valor de mercado actual (numérico)': 'Valor de mercado',
        'overallrating': 'Overall',
        'potential': 'Potencial',
        'Nombre completo': 'Nombre',
        'Club actual': 'Club',
        'Posición principal': 'Posición'
    };

    sortSelect.innerHTML = '';
    
    for (const [field, label] of Object.entries(commonSortFields)) {
        if (csvHeaders.includes(field)) {
            const option = document.createElement('option');
            option.value = field;
            option.textContent = label;
            sortSelect.appendChild(option);
        }
    }
}

function performSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showMessage(searchError, 'Por favor, ingresa un término de búsqueda', true);
        return;
    }
    
    if (!allPlayers || allPlayers.length === 0) {
        showMessage(searchError, 'No hay datos cargados para buscar', true);
        return;
    }
    
    showMessage(searchError, '', false);
    searchLoading.style.display = 'block';
    
    setTimeout(() => {
        try {
            currentResults = smartSearch(query, allPlayers);
            
            searchLoading.style.display = 'none';
            
            if (currentResults.length === 0) {
                showMessage(searchError, 'No se encontraron resultados para tu búsqueda', true);
                resultsSection.style.display = 'none';
            } else {
                displayResults(currentResults);
                resultsSection.style.display = 'block';
            }
        } catch (error) {
            searchLoading.style.display = 'none';
            showMessage(searchError, 'Error en la búsqueda: ' + error.message, true);
        }
    }, 100);
}

function processQuery(query) {
    // Búsqueda de jugador específico
    if (!query.includes('más') && !query.includes('menos') && !query.includes('>') && !query.includes('<')) {
        return playersData.filter(player => {
            const searchFields = ['Nombre completo', 'Club actual', 'Posición principal'];
            return searchFields.some(field => {
                const value = player[field];
                return value && value.toString().toLowerCase().includes(query);
            });
        });
    }

    // Top jugadores más valiosos
    if (query.includes('más valiosos') || query.includes('top') && query.includes('valiosos')) {
        const num = extractNumber(query) || 10;
        return playersData
            .filter(p => p['Valor de mercado actual (numérico)'] && p['Valor de mercado actual (numérico)'] > 0)
            .sort((a, b) => (b['Valor de mercado actual (numérico)'] || 0) - (a['Valor de mercado actual (numérico)'] || 0))
            .slice(0, num);
    }

    // Top jugadores menos valiosos
    if (query.includes('menos valiosos')) {
        const num = extractNumber(query) || 10;
        return playersData
            .filter(p => p['Valor de mercado actual (numérico)'] && p['Valor de mercado actual (numérico)'] > 0)
            .sort((a, b) => (a['Valor de mercado actual (numérico)'] || 0) - (b['Valor de mercado actual (numérico)'] || 0))
            .slice(0, num);
    }

    // Filtros por atributos (ej: overall > 90)
    const conditionMatch = query.match(/(\w+)\s*([><=]+)\s*(\d+)/);
    if (conditionMatch) {
        const [, field, operator, value] = conditionMatch;
        const numValue = parseInt(value);
        
        // Mapear nombres de campos comunes
        let actualField = field;
        if (field === 'overall') actualField = 'overallrating';
        if (field === 'potential') actualField = 'potential';
        if (field === 'valor') actualField = 'Valor de mercado actual (numérico)';
        
        return playersData.filter(player => {
            const playerValue = player[actualField];
            if (playerValue === undefined || playerValue === null) return false;
            
            switch (operator) {
                case '>': return playerValue > numValue;
                case '<': return playerValue < numValue;
                case '>=': return playerValue >= numValue;
                case '<=': return playerValue <= numValue;
                case '=': return playerValue == numValue;
                default: return false;
            }
        });
    }

    // Búsqueda por posición
    if (query.includes('portero') || query.includes('gk')) {
        return playersData.filter(p => 
            p['Posición principal'] && p['Posición principal'].toLowerCase().includes('portero')
        );
    }

    // Búsqueda general
    return playersData.filter(player => {
        return Object.values(player).some(value => 
            value && value.toString().toLowerCase().includes(query)
        );
    });
}

function extractNumber(text) {
    const match = text.match(/\d+/);
    return match ? parseInt(match[0]) : null;
}

function displayResults(data) {
    if (data.length === 0) {
        resultsSection.style.display = 'block';
        resultsCount.textContent = 'No se encontraron resultados';
        resultsGrid.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); font-size: 1rem; padding: 3rem;">No hay jugadores que coincidan con tu búsqueda</div>';
        return;
    }

    filteredData = data;
    resultsSection.style.display = 'block';
    resultsCount.textContent = `${data.length} jugador${data.length !== 1 ? 'es' : ''} encontrado${data.length !== 1 ? 's' : ''}`;
    
    resultsGrid.innerHTML = '';
    data.forEach(player => {
        const card = createPlayerCard(player);
        resultsGrid.appendChild(card);
    });
}

// Función auxiliar para truncar texto y agregar tooltip
function createTruncatedElement(text, maxLength = 20, elementType = 'span') {
    const element = document.createElement(elementType);
    
    if (text && text.length > maxLength) {
        element.className = 'truncate-text';
        element.textContent = text.substring(0, maxLength) + '...';
        element.setAttribute('data-full-text', text);
        element.title = text; // Fallback para navegadores que no soporten CSS hover
    } else {
        element.textContent = text || '';
    }
    
    return element;
}

function createPlayerCard(player) {
    const card = document.createElement('div');
    card.className = 'player-card';
    card.onclick = () => showPlayerDetail(player);

    const name = player['Nombre completo'] || 'Nombre no disponible';
    const value = formatCurrency(player['Valor de mercado actual (numérico)']);
    const predictedValue = player['Valor_Predicho'] || 0;
    const currentValue = player['Valor de mercado actual (numérico)'] || 0;
    const overall = player['overallrating'] || 'N/A';
    const potential = player['potential'] || 'N/A';
    const position = player['Posición principal'] || 'N/A';
    const club = player['Club actual'] || 'N/A';

    // Determinar color del valor predicho
    let predictedValueClass = 'predicted-value-neutral';
    if (predictedValue > currentValue) {
        predictedValueClass = 'predicted-value-higher';
    } else if (predictedValue < currentValue) {
        predictedValueClass = 'predicted-value-lower';
    }

    card.innerHTML = `
        <div class="player-name">${name}</div>
        <div class="player-value-container">
            <div class="player-value">${value}</div>
            <div class="predicted-value ${predictedValueClass}">Predicho: ${formatCurrency(predictedValue)}</div>
        </div>
        <div class="player-stats">
            <div class="stat-item">
                <span>Overall</span>
                <span>${overall}</span>
            </div>
            <div class="stat-item">
                <span>Potencial</span>
                <span>${potential}</span>
            </div>
            <div class="stat-item">
                <span>Posición</span>
                <span>${position}</span>
            </div>
            <div class="stat-item">
                <span>Club</span>
                <span title="${club}">${club}</span>
            </div>
        </div>
    `;

    return card;
}

function formatCurrency(value) {
    if (!value || value === 0) return 'No valorado';
    
    if (value >= 1000000) {
        return '€' + (value / 1000000).toFixed(1) + 'M';
    } else if (value >= 1000) {
        return '€' + (value / 1000).toFixed(0) + 'K';
    } else {
        return '€' + value.toLocaleString();
    }
}

function sortResults() {
    const sortField = sortSelect.value;
    if (!filteredData.length) return;

    filteredData.sort((a, b) => {
        const aVal = a[sortField];
        const bVal = b[sortField];

        if (typeof aVal === 'string' && typeof bVal === 'string') {
            return aVal.localeCompare(bVal);
        } else {
            return (bVal || 0) - (aVal || 0);
        }
    });

    displayResults(filteredData);
}

function showPlayerDetail(player) {
    const playerName = player['Nombre completo'] || 'Nombre no disponible';
    const currentValue = player['Valor de mercado actual (numérico)'] || 0;
    const predictedValue = player['Valor_Predicho'] || 0;
    
    document.getElementById('modal-player-name').textContent = playerName;
    
    // Actualizar el contenedor de valor en el modal
    const modalValueElement = document.getElementById('modal-player-value');
    modalValueElement.innerHTML = `
        <div class="modal-value-current">${formatCurrency(currentValue)}</div>
        <div class="modal-value-predicted ${predictedValue > currentValue ? 'predicted-higher' : predictedValue < currentValue ? 'predicted-lower' : 'predicted-neutral'}">
            Predicho: ${formatCurrency(predictedValue)}
        </div>
    `;
    
    // Llenar estadísticas detalladas
    const detailedStats = document.getElementById('detailed-stats');
    detailedStats.innerHTML = '';
    
    let importantStats;
    const position = player['Posición principal'] || '';
    
    // Definir estadísticas según la posición del jugador
    if (position.toLowerCase().includes('portero')) {
        // Estadísticas específicas para porteros (coinciden con el gráfico radar)
        importantStats = {
            // Información básica primero
            'Valor de mercado actual (numérico)': 'Valor de mercado',
            'Valor_Predicho': 'Valor predicho',
            'Club actual': 'Club',
            'Posición principal': 'Posición',
            'Nacionalidad': 'Nacionalidad',
            'Lugar de nacimiento (país)': 'País de nacimiento',
            
            // Atributos técnicos específicos de portero
            'overallrating': 'Overall',
            'potential': 'Potencial',
            'gk_kicking': 'Saque',
            'gk_reflexes': 'Reflejos',
            'gk_diving': 'Estirada',
            'gk_handling': 'Manejo',
            'gk_positioning': 'Posicionamiento',
            'reactions': 'Reacciones'
        };
    } else {
        // Estadísticas para jugadores de campo
        importantStats = {
            // Información básica primero
            'Valor de mercado actual (numérico)': 'Valor de mercado',
            'Valor_Predicho': 'Valor predicho',
            'Club actual': 'Club',
            'Posición principal': 'Posición',
            'Nacionalidad': 'Nacionalidad',
            'Lugar de nacimiento (país)': 'País de nacimiento',
            
            // Atributos técnicos después
            'overallrating': 'Overall',
            'potential': 'Potencial',
            'acceleration': 'Aceleración',
            'ballcontrol': 'Control de balón',
            'dribbling': 'Regate',
            'finishing': 'Definición',
            'longpassing': 'Pase largo',
            'reactions': 'Reacciones',
            'shotpower': 'Potencia de tiro',
            'strength': 'Fuerza',
            'vision': 'Visión'
        };
    }

    for (const [key, label] of Object.entries(importantStats)) {
        if (player[key] !== undefined && player[key] !== null && player[key] !== '') {
            const row = document.createElement('div');
            row.className = 'stat-row';
            
            let displayValue = player[key];
            if (key === 'Valor de mercado actual (numérico)' || key === 'Valor_Predicho') {
                displayValue = formatCurrency(player[key]);
            }
            
            const labelSpan = document.createElement('span');
            labelSpan.className = 'stat-label';
            labelSpan.textContent = label;
            
            const valueSpan = document.createElement('span');
            valueSpan.className = 'stat-value';
            
            // Aplicar truncamiento a valores largos
            if (typeof displayValue === 'string' && displayValue.length > 25) {
                valueSpan.className = 'stat-value truncate-text';
                valueSpan.textContent = displayValue.substring(0, 25) + '...';
                valueSpan.setAttribute('data-full-text', displayValue);
                valueSpan.title = displayValue;
            } else {
                valueSpan.textContent = displayValue;
            }
            
            row.appendChild(labelSpan);
            row.appendChild(valueSpan);
            detailedStats.appendChild(row);
        }
    }

    // Crear gráfico radar
    createRadarChart(player);
    
    playerModal.style.display = 'block';
    playerModal.setAttribute('aria-hidden', 'false');
}

function createRadarChart(player) {
    const position = player['Posición principal'] || '';
    let radarData = {};
    let spanishLabels = {};
    
    // Configurar estadísticas según la posición
    if (position.toLowerCase().includes('portero')) {
        // Porteros - eliminar tiros lejanos y potencia
        radarData = {
            gk_kicking: player.gk_kicking || 0,
            gk_reflexes: player.gk_reflexes || 0,
            gk_diving: player.gk_diving || 0,
            gk_handling: player.gk_handling || 0,
            gk_positioning: player.gk_positioning || 0,
            reactions: player.reactions || 0
        };
        
        spanishLabels = {
            gk_kicking: 'Saque',
            gk_reflexes: 'Reflejos',
            gk_diving: 'Estirada',
            gk_handling: 'Manejo',
            gk_positioning: 'Posicionamiento',
            reactions: 'Reacciones'
        };
    } else if (position.toLowerCase().includes('delantero') || position.toLowerCase().includes('extremo')) {
        // Delanteros
        radarData = {
            overallrating: player.overallrating || 0,
            potential: player.potential || 0,
            strength: player.strength || 0,
            agility: player.agility || 0,
            headingaccuracy: player.headingaccuracy || 0,
            longshots: player.longshots || 0,
            ballcontrol: player.ballcontrol || 0,
            acceleration: player.acceleration || 0,
            vision: player.vision || 0
        };
        
        spanishLabels = {
            overallrating: 'Overall',
            potential: 'Potencial',
            strength: 'Fuerza',
            agility: 'Agilidad',
            headingaccuracy: 'Cabeceo',
            longshots: 'Tiros lejanos',
            ballcontrol: 'Control',
            acceleration: 'Aceleración',
            vision: 'Visión'
        };
    } else if (position.toLowerCase().includes('defensa') || position.toLowerCase().includes('lateral')) {
        // Defensas
        radarData = {
            potential: player.potential || 0,
            jumping: player.jumping || 0,
            acceleration: player.acceleration || 0,
            headingaccuracy: player.headingaccuracy || 0,
            strength: player.strength || 0,
            ballcontrol: player.ballcontrol || 0,
            standingtackle: player.standingtackle || 0
        };
        
        spanishLabels = {
            potential: 'Potencial',
            jumping: 'Salto',
            acceleration: 'Aceleración',
            headingaccuracy: 'Cabeceo',
            strength: 'Fuerza',
            ballcontrol: 'Control',
            standingtackle: 'Entrada'
        };
    } else {
        // Centrocampistas (por defecto)
        radarData = {
            overallrating: player.overallrating || 0,
            potential: player.potential || 0,
            strength: player.strength || 0,
            vision: player.vision || 0,
            ballcontrol: player.ballcontrol || 0,
            acceleration: player.acceleration || 0,
            interceptions: player.interceptions || 0,
            reactions: player.reactions || 0,
            longpassing: player.longpassing || 0,
            dribbling: player.dribbling || 0,
            standingtackle: player.standingtackle || 0
        };
        
        spanishLabels = {
            overallrating: 'Overall',
            potential: 'Potencial',
            strength: 'Fuerza',
            vision: 'Visión',
            ballcontrol: 'Control',
            acceleration: 'Aceleración',
            interceptions: 'Intercepciones',
            reactions: 'Reacciones',
            longpassing: 'Pase largo',
            dribbling: 'Regate',
            standingtackle: 'Entrada'
        };
    }

    // Limpiar el gráfico anterior
    d3.select('#radar-chart').selectAll('*').remove();

    const width = 280;
    const height = 280;
    const margin = 40;
    const radius = Math.min(width, height) / 2 - margin;

    const svg = d3.select('#radar-chart')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${width/2}, ${height/2})`);

    // Configuración del radar
    const attributes = Object.keys(radarData);
    const angleSlice = Math.PI * 2 / attributes.length;

    // Escalas - ajustar según el tipo de estadística
    const maxValue = Math.max(...Object.values(radarData), 100);
    const rScale = d3.scaleLinear()
        .domain([0, maxValue])
        .range([0, radius]);

    // Líneas de fondo
    const levels = 5;
    for (let i = 1; i <= levels; i++) {
        const levelRadius = radius * i / levels;
        g.append('circle')
            .attr('r', levelRadius)
            .style('fill', 'none')
            .style('stroke', '#e5e7eb')
            .style('stroke-width', '1px');
    }

    // Líneas radiales
    attributes.forEach((attr, i) => {
        g.append('line')
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('x2', rScale(maxValue) * Math.cos(angleSlice * i - Math.PI / 2))
            .attr('y2', rScale(maxValue) * Math.sin(angleSlice * i - Math.PI / 2))
            .style('stroke', '#e5e7eb')
            .style('stroke-width', '1px');
    });

    // Etiquetas en español
    attributes.forEach((attr, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        const x = (rScale(maxValue) + 20) * Math.cos(angle);
        const y = (rScale(maxValue) + 20) * Math.sin(angle);
        
        g.append('text')
            .attr('x', x)
            .attr('y', y)
            .attr('dy', '0.35em')
            .style('font-size', '10px')
            .style('fill', '#6b7280')
            .style('text-anchor', 'middle')
            .style('font-weight', '500')
            .text(spanishLabels[attr] || attr);
    });

    // Datos del jugador
    const line = d3.line()
        .x((d, i) => rScale(d.value) * Math.cos(angleSlice * i - Math.PI / 2))
        .y((d, i) => rScale(d.value) * Math.sin(angleSlice * i - Math.PI / 2))
        .curve(d3.curveCardinalClosed);

    const playerData = attributes.map(attr => ({
        attribute: attr,
        value: radarData[attr]
    }));

    g.append('path')
        .datum(playerData)
        .attr('d', line)
        .style('fill', 'rgba(37, 99, 235, 0.2)')
        .style('stroke', '#2563eb')
        .style('stroke-width', '2px');

    // Puntos
    g.selectAll('.radar-dot')
        .data(playerData)
        .enter()
        .append('circle')
        .attr('class', 'radar-dot')
        .attr('cx', (d, i) => rScale(d.value) * Math.cos(angleSlice * i - Math.PI / 2))
        .attr('cy', (d, i) => rScale(d.value) * Math.sin(angleSlice * i - Math.PI / 2))
        .attr('r', 3)
        .style('fill', '#2563eb')
        .style('stroke', '#ffffff')
        .style('stroke-width', '2px');
}

function closeModal() {
    playerModal.style.display = 'none';
    playerModal.setAttribute('aria-hidden', 'true');
}

function showMessage(element, message, show) {
    if (show) {
        element.textContent = message;
        element.style.display = 'block';
    } else {
        element.style.display = 'none';
    }
}

// Función para mostrar la tabla completa
function showTable() {
    searchSection.style.display = 'none';
    resultsSection.style.display = 'none';
    tableSection.style.display = 'block';
    
    // Inicializar filtros
    initializeFilters();
    
    // Crear headers de la tabla
    createTableHeaders();
    
    // Aplicar límite inicial
    applyTableLimit();
    
    // Renderizar tabla
    renderTable();
}

// Función para inicializar filtros
function initializeFilters() {
    // Inicializar filtros de precio
    initializePriceFilters();
    
    // Inicializar filtros de posición
    initializePositionFilters();
    
    // Agregar event listeners
    setupFilterEventListeners();
}

// Función para inicializar filtros de precio
function initializePriceFilters() {
    const currentValues = allPlayers
        .map(p => parseFloat(p['Valor de mercado actual (numérico)']) || 0)
        .filter(v => v > 0);
    
    const predictedValues = allPlayers
        .map(p => parseFloat(p['Valor_Predicho']) || 0)
        .filter(v => v > 0);
    
    const maxCurrent = Math.max(...currentValues) / 1000000; // Convertir a millones
    const maxPredicted = Math.max(...predictedValues) / 1000000;
    const maxValue = Math.max(maxCurrent, maxPredicted, 200);
    
    // Actualizar sliders
    const sliders = ['min-current-slider', 'max-current-slider', 'min-predicted-slider', 'max-predicted-slider'];
    sliders.forEach(id => {
        const slider = document.getElementById(id);
        if (slider) {
            slider.max = Math.ceil(maxValue);
            if (id.includes('max')) {
                slider.value = Math.ceil(maxValue);
            }
        }
    });
    
    // Actualizar valores mostrados
    updatePriceDisplays();
}

// Función para inicializar filtros de posición
function initializePositionFilters() {
    const positions = [...new Set(allPlayers
        .map(p => p['Posición principal'])
        .filter(p => p && p.trim() !== '')
    )].sort();
    
    const positionFiltersContainer = document.getElementById('position-filters');
    positionFiltersContainer.innerHTML = '';
    
    positions.forEach(position => {
        const item = document.createElement('div');
        item.className = 'position-filter-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `pos-${position.replace(/\s+/g, '-')}`;
        checkbox.value = position;
        checkbox.addEventListener('change', applyFilters);
        
        const label = document.createElement('label');
        label.htmlFor = checkbox.id;
        label.textContent = position;
        
        item.appendChild(checkbox);
        item.appendChild(label);
        positionFiltersContainer.appendChild(item);
    });
}

// Función para configurar event listeners de filtros
function setupFilterEventListeners() {
    // Sliders de precio
    ['min-current-slider', 'max-current-slider', 'min-predicted-slider', 'max-predicted-slider'].forEach(id => {
        const slider = document.getElementById(id);
        if (slider) {
            slider.addEventListener('input', () => {
                updatePriceDisplays();
                applyFilters();
            });
        }
    });
    
    // Botón limpiar posiciones
    const clearBtn = document.getElementById('clear-position-filters');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            document.querySelectorAll('.position-filter-item input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });
            applyFilters();
        });
    }
}

// Función para actualizar displays de precio
function updatePriceDisplays() {
    const displays = {
        'min-current-value': 'min-current-slider',
        'max-current-value': 'max-current-slider',
        'min-predicted-value': 'min-predicted-slider',
        'max-predicted-value': 'max-predicted-slider'
    };
    
    Object.entries(displays).forEach(([displayId, sliderId]) => {
        const display = document.getElementById(displayId);
        const slider = document.getElementById(sliderId);
        if (display && slider) {
            display.textContent = slider.value;
        }
    });
}

// Función para aplicar todos los filtros
function applyFilters() {
    // Obtener valores de filtros de precio
    const minCurrent = parseFloat(document.getElementById('min-current-slider')?.value || 0) * 1000000;
    const maxCurrent = parseFloat(document.getElementById('max-current-slider')?.value || 200) * 1000000;
    const minPredicted = parseFloat(document.getElementById('min-predicted-slider')?.value || 0) * 1000000;
    const maxPredicted = parseFloat(document.getElementById('max-predicted-slider')?.value || 200) * 1000000;
    
    // Obtener posiciones seleccionadas
    const selectedPositions = Array.from(document.querySelectorAll('.position-filter-item input[type="checkbox"]:checked'))
        .map(cb => cb.value);
    
    // Aplicar filtros
    let filtered = [...allPlayers];
    
    // Filtro de texto
    const filterText = document.getElementById('table-filter')?.value.toLowerCase().trim();
    if (filterText) {
        filtered = filtered.filter(player => {
            return tableColumns.some(column => {
                const value = player[column.key];
                return value && value.toString().toLowerCase().includes(filterText);
            });
        });
    }
    
    // Filtros de precio
    filtered = filtered.filter(player => {
        const currentValue = parseFloat(player['Valor de mercado actual (numérico)']) || 0;
        const predictedValue = parseFloat(player['Valor_Predicho']) || 0;
        
        return currentValue >= minCurrent && currentValue <= maxCurrent &&
               predictedValue >= minPredicted && predictedValue <= maxPredicted;
    });
    
    // Filtro de posiciones
    if (selectedPositions.length > 0) {
        filtered = filtered.filter(player => {
            const position = player['Posición principal'];
            return position && selectedPositions.includes(position);
        });
    }
    
    // Aplicar límite
    const limit = document.getElementById('limit-select')?.value;
    if (limit && limit !== 'all') {
        filtered = filtered.slice(0, parseInt(limit));
    }
    
    filteredTableData = filtered;
    renderTable();
}

// Función para renderizar tabla (actualizada)
function renderTable() {
    tableBody.innerHTML = '';
    
    // Actualizar información de la tabla
    const totalPlayers = allPlayers.length;
    const showingPlayers = filteredTableData.length;
    const limitText = currentTableLimit === 'all' ? 'todos' : currentTableLimit;
    
    tableInfo.textContent = `Mostrando ${showingPlayers} de ${totalPlayers} jugadores`;
    
    // Renderizar filas
    filteredTableData.forEach(player => {
        const row = document.createElement('tr');
        
        tableColumns.forEach(column => {
            const cell = document.createElement('td');
            const value = player[column.key];
            
            if (column.type === 'currency') {
                cell.textContent = formatCurrency(value);
                cell.classList.add('table-cell-value');
            } else if (column.type === 'predicted') {
                const currentValue = parseFloat(player['Valor de mercado actual (numérico)']) || 0;
                const predictedValue = parseFloat(value) || 0;
                
                cell.textContent = formatCurrency(predictedValue);
                
                if (predictedValue > currentValue) {
                    cell.classList.add('table-cell-predicted-higher');
                } else if (predictedValue < currentValue) {
                    cell.classList.add('table-cell-predicted-lower');
                } else {
                    cell.classList.add('table-cell-predicted-neutral');
                }
            } else if (column.type === 'difference_colored') {
                const currentValue = parseFloat(player['Valor de mercado actual (numérico)']) || 0;
                const predictedValue = parseFloat(player['Valor_Predicho']) || 0;
                const diffValue = predictedValue - currentValue;
                
                cell.textContent = formatCurrency(Math.abs(diffValue));
                
                if (predictedValue > currentValue) {
                    cell.classList.add('table-cell-positive');
                } else if (predictedValue < currentValue) {
                    cell.classList.add('table-cell-negative');
                } else {
                    cell.classList.add('table-cell-predicted-neutral');
                }
            } else if (column.type === 'percentage') {
                const percentValue = parseFloat(value) || 0;
                // El valor ya está en porcentaje, solo redondear a máximo 3 decimales
                const cleanPercentage = parseFloat(percentValue.toFixed(3)).toString();
                cell.textContent = cleanPercentage + '%';
                
                if (percentValue > 0) {
                    cell.classList.add('table-cell-positive');
                } else if (percentValue < 0) {
                    cell.classList.add('table-cell-negative');
                } else {
                    cell.classList.add('table-cell-predicted-neutral');
                }
            } else {
                cell.textContent = value || 'N/A';
                if (column.key === 'Nombre completo') {
                    cell.classList.add('table-cell-name');
                }
            }
            
            row.appendChild(cell);
        });
        
        // Agregar evento click para mostrar detalles
        row.addEventListener('click', () => showPlayerDetail(player));
        
        tableBody.appendChild(row);
    });
}

// Actualizar función filterTable para usar la nueva lógica
function filterTable() {
    applyFilters();
}

// Actualizar función applyTableLimit
function applyTableLimit() {
    const limit = limitSelect.value;
    currentTableLimit = limit;
    applyFilters();
}

// Función para volver a la búsqueda
function backToSearch() {
    tableSection.style.display = 'none';
    searchSection.style.display = 'block';
    resultsSection.style.display = 'block';
}

// Función para crear headers de la tabla
function createTableHeaders() {
    tableHeaders.innerHTML = '';
    
    tableColumns.forEach(column => {
        const th = document.createElement('th');
        th.textContent = column.label;
        th.dataset.column = column.key;
        
        if (column.sortable) {
            th.classList.add('sortable');
            th.addEventListener('click', () => sortTable(column.key));
        }
        
        tableHeaders.appendChild(th);
    });
}

// Función para ordenar tabla
function sortTable(columnKey) {
    // Actualizar dirección de ordenamiento
    if (currentSortColumn === columnKey) {
        currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        currentSortColumn = columnKey;
        currentSortDirection = 'asc';
    }
    
    // Actualizar clases de los headers
    document.querySelectorAll('.players-table th').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
    });
    
    const currentHeader = document.querySelector(`[data-column="${columnKey}"]`);
    if (currentHeader) {
        currentHeader.classList.add(currentSortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
    }
    
    // Ordenar datos
    filteredTableData.sort((a, b) => {
        let valueA = a[columnKey];
        let valueB = b[columnKey];
        
        // Convertir a números si es necesario
        if (!isNaN(valueA) && !isNaN(valueB)) {
            valueA = parseFloat(valueA) || 0;
            valueB = parseFloat(valueB) || 0;
        } else {
            valueA = (valueA || '').toString().toLowerCase();
            valueB = (valueB || '').toString().toLowerCase();
        }
        
        let comparison = 0;
        if (valueA > valueB) comparison = 1;
        if (valueA < valueB) comparison = -1;
        
        return currentSortDirection === 'asc' ? comparison : -comparison;
    });
    
    renderTable();
}

// Event Listeners principales
document.addEventListener('DOMContentLoaded', function() {
    loadData();
    
    // Búsqueda
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    // Búsquedas rápidas
    document.querySelectorAll('.quick-search-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const query = this.dataset.query;
            searchInput.value = query;
            performSearch();
        });
    });
    
    // Ordenamiento
    sortSelect.addEventListener('change', function() {
        if (currentResults.length > 0) {
            sortResults(this.value);
        }
    });
    
    // Modal
    modalClose.addEventListener('click', closeModal);
    playerModal.addEventListener('click', function(e) {
        if (e.target === playerModal) {
            closeModal();
        }
    });
    
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
    
    // Tabla
    tableBtn.addEventListener('click', showTable);
    backToSearchBtn.addEventListener('click', backToSearch);
    
    limitSelect.addEventListener('change', function() {
        applyTableLimit();
        filterTable();
    });
    
    tableFilter.addEventListener('input', filterTable);
});

// Función de búsqueda inteligente
function smartSearch(query, data) {
    const lowerQuery = query.toLowerCase();
    
    // Búsqueda por nombre exacto
    let results = data.filter(player => {
        const name = player['Nombre completo'];
        return name && name.toLowerCase().includes(lowerQuery);
    });
    
    // Si no hay resultados por nombre, buscar por club
    if (results.length === 0) {
        results = data.filter(player => {
            const club = player['Club actual'];
            return club && club.toLowerCase().includes(lowerQuery);
        });
    }
    
    // Si aún no hay resultados, buscar por posición
    if (results.length === 0) {
        results = data.filter(player => {
            const position = player['Posición principal'];
            return position && position.toLowerCase().includes(lowerQuery);
        });
    }
    
    // Búsqueda por criterios numéricos (ej: "overallrating > 90")
    if (results.length === 0) {
        const numericMatch = lowerQuery.match(/(\w+)\s*([><]=?)\s*(\d+)/);
        if (numericMatch) {
            const [, field, operator, value] = numericMatch;
            const numValue = parseFloat(value);
            
            results = data.filter(player => {
                const playerValue = parseFloat(player[field]);
                if (isNaN(playerValue)) return false;
                
                switch (operator) {
                    case '>': return playerValue > numValue;
                    case '>=': return playerValue >= numValue;
                    case '<': return playerValue < numValue;
                    case '<=': return playerValue <= numValue;
                    default: return false;
                }
            });
        }
    }
    
    return results;
}

// Función para mostrar resultados
function displayResults(results) {
    resultsCount.textContent = `${results.length} jugador${results.length !== 1 ? 'es' : ''} encontrado${results.length !== 1 ? 's' : ''}`;
    
    resultsGrid.innerHTML = '';
    
    results.forEach(player => {
        const card = createPlayerCard(player);
        resultsGrid.appendChild(card);
    });
}

// Función para ordenar resultados
function sortResults(sortBy) {
    if (!currentResults || currentResults.length === 0) return;
    
    const sorted = [...currentResults].sort((a, b) => {
        let valueA, valueB;
        
        switch (sortBy) {
            case 'value_eur':
                valueA = parseFloat(a['Valor de mercado actual (numérico)']) || 0;
                valueB = parseFloat(b['Valor de mercado actual (numérico)']) || 0;
                return valueB - valueA; // Descendente
            case 'overall':
                valueA = parseFloat(a['overallrating']) || 0;
                valueB = parseFloat(b['overallrating']) || 0;
                return valueB - valueA; // Descendente
            case 'age':
                valueA = parseFloat(a['Edad']) || 0;
                valueB = parseFloat(b['Edad']) || 0;
                return valueA - valueB; // Ascendente
            case 'short_name':
                valueA = a['Nombre completo'] || '';
                valueB = b['Nombre completo'] || '';
                return valueA.localeCompare(valueB); // Ascendente
            default:
                return 0;
        }
    });
    
    displayResults(sorted);
}
