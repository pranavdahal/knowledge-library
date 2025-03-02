import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Neo4j Graph Visualization", layout="wide")
st.info("Click on a node to view details and expand connections.")

html = """
<!DOCTYPE html>
<html>
<head>
  <title>Knowledge Graph Visualization</title>
  <style>
    body { 
      font-family: Arial, sans-serif; 
      margin: 0; 
      padding: 0; 
    }
    .container { 
      display: flex; 
      height: 100vh; 
    }
    #cy { 
      flex: 1; 
      height: 100%;
      width: 80%; 
      border: 1px solid #ddd;
    }
    #node-details { 
      width: 20%; 
      padding: 20px; 
      background: #f5f5f5; 
      border: 1px solid #ddd;
      overflow-y: auto;
      display: none;
    }
    
    .node-controls {
      margin-top: 20px;
      padding: 10px;
      background-color: #f0f0f0;
      border-radius: 5px;
    }
    
    .loading {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background-color: rgba(255, 255, 255, 0.8);
      padding: 20px;
      border-radius: 5px;
      box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
  </style>
  <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/neo4j-driver"></script>
</head>
<body>
  <div class="container" style="display: flex; justify-content: center; width: 100%;">
    <div id="cy"></div>
    <div id="node-details"></div>
  </div>
  
  <div id="loading" class="loading" style="display: none;">
    Loading graph data...
  </div>
  
  <script>
    const cy = cytoscape({
    container: document.getElementById('cy'),
    style: [
        {
        selector: 'node',
        style: {
            'label': 'data(label)',
            'background-color': '#6FB1FC', // Default color
            'width': 60,
            'height': 60,
            'text-valign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': '80px',
            'font-size': '10px'
        }
        },
        {
        selector: 'node[type="qna"]',
        style: {
            'background-color': '#03a9f4', // Blue
            'shape': 'roundrectangle'
        }
        },
        {
        selector: 'node[type="category"]',
        style: {
            'background-color': '#4caf50', // Green
            'shape': 'roundrectangle'
        }
        },
        {
        selector: 'node[type="subcategory"]',
        style: {
            'background-color': '#ff9800', // Orange
            'shape': 'ellipse'
        }
        },
        {
        selector: 'node[type="control"]',
        style: {
            'background-color': '#e91e63', // Pink
            'shape': 'diamond'
        }
        },
        {
        selector: 'node[type="domain"]',
        style: {
            'background-color': '#9c27b0', // Purple
            'shape': 'rectangle'
        }
        },
        {
        selector: 'node[type="compliancestandard"]',
        style: {
            'background-color': '#795548', // Brown
            'shape': 'round-triangle'
        }
        },
        {
        selector: 'edge',
        style: {
            'label': 'data(relationship)',
            'width': 2,
            'curve-style': 'bezier',
            'line-color': '#ccc',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#ccc',
            'text-rotation': 'autorotate',
            'font-size': '8px'
        }
        }
    ]
    });

    async function buildGraphFromNeo4j() {
      // Show loading indicator
      document.getElementById('loading').style.display = 'block';
      
      const driver = neo4j.driver(
        'neo4j+s://c282989d.databases.neo4j.io',
        neo4j.auth.basic('neo4j', 'tIQavAzq7h_7foQtpPRnDBGBdoSL8QHMroRrNs9u8dE')
      );
      
      const session = driver.session();
      
      try {
        // Get a more meaningful subset of the graph
        const result = await session.run(
          'MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50'
        );
        
        const nodeMap = new Map(); // To track added nodes
        const elements = [];
        
        result.records.forEach(record => {
          const source = record.get('n');
          const target = record.get('m');
          const relationship = record.get('r');
          
          // Add source node if not already added
          if (!nodeMap.has(source.identity.toString())) {
            nodeMap.set(source.identity.toString(), true);
            elements.push({
              data: {
                id: source.identity.toString(),
                label: source.properties.name || source.properties.text || source.labels[0],
                type: source.labels[0].toLowerCase(),
                properties: source.properties
              }
            });
          }
          
          // Add target node if not already added
          if (!nodeMap.has(target.identity.toString())) {
            nodeMap.set(target.identity.toString(), true);
            elements.push({
              data: {
                id: target.identity.toString(),
                label: target.properties.name || target.properties.text || target.labels[0],
                type: target.labels[0].toLowerCase(),
                properties: target.properties
              }
            });
          }
          
          // Add relationship (edge)
          elements.push({
            data: {
              id: relationship.identity.toString(),
              source: source.identity.toString(),
              target: target.identity.toString(),
              relationship: relationship.type,
              properties: relationship.properties
            }
          });
        });
        
        cy.add(elements);
        
        // Apply a layout appropriate for knowledge graphs
        cy.layout({
          name: 'cose',
          idealEdgeLength: 100,
          nodeOverlap: 20,
          refresh: 20,
          fit: true,
          padding: 30,
          randomize: false,
          componentSpacing: 100,
          nodeRepulsion: 400000,
          edgeElasticity: 100,
          nestingFactor: 5,
          gravity: 80,
          numIter: 1000,
          initialTemp: 200,
          coolingFactor: 0.95,
          minTemp: 1.0
        }).run();
        
      } catch (error) {
        console.error("Error connecting to Neo4j:", error);
        document.getElementById('node-details').innerHTML = 
          `<h3>Error</h3><p>Could not connect to Neo4j database: ${error.message}</p>`;
        document.getElementById('node-details').style.display = 'block';
      } finally {
        await session.close();
        await driver.close();
        // Hide loading indicator
        document.getElementById('loading').style.display = 'none';
      }
    }
    
    cy.on('tap', 'node', function(evt) {
      const node = evt.target;
      const properties = node.data('properties');
      const nodeType = node.data('type');
      
      displayNodeDetails(node.id(), properties, nodeType);
    });
    
    function displayNodeDetails(id, properties, nodeType) {
      const detailsDiv = document.getElementById('node-details');
      
      let html = `<h3>Node Details: ${nodeType || 'Unknown'}</h3>`;
      
      if (nodeType === 'question') {
        html += `<div class="question-text">${properties.text || 'No question text'}</div>`;
      } else if (nodeType === 'answer') {
        html += `<div class="answer-text">${properties.text || 'No answer text'}</div>`;
      }
      
      html += '<h4>Properties:</h4><ul>';
      
      for (const [key, value] of Object.entries(properties)) {
        html += `<li><strong>${key}:</strong> ${value}</li>`;
      }
      
      html += '</ul>';
      
      // Add controls for the selected node
      html += `
        <div class="node-controls">
          <button onclick="expandNode('${id}')">Expand Connections</button>
          <button onclick="hideNode('${id}')">Hide Node</button>
        </div>
      `;
      
      detailsDiv.innerHTML = html;
      detailsDiv.style.display = 'block';
    }
    
    // Function to expand node connections
    async function expandNode(nodeId) {
      const driver = neo4j.driver(
        'neo4j+s://c282989d.databases.neo4j.io',
        neo4j.auth.basic('neo4j', 'tIQavAzq7h_7foQtpPRnDBGBdoSL8QHMroRrNs9u8dE')
      );
      
      const session = driver.session();
      
      try {
        // Get connections for the specified node
        const result = await session.run(
          `MATCH (n)-[r]-(m) WHERE ID(n) = ${nodeId} RETURN n, r, m`
        );
        
        const nodeMap = new Map(); // To track added nodes
        const elements = [];
        
        result.records.forEach(record => {
          const source = record.get('n');
          const target = record.get('m');
          const relationship = record.get('r');
          
          // Check if target node already exists in the graph
          if (!cy.getElementById(target.identity.toString()).length) {
            elements.push({
              data: {
                id: target.identity.toString(),
                label: target.properties.name || target.properties.text || target.labels[0],
                type: target.labels[0].toLowerCase(),
                properties: target.properties
              }
            });
          }
          
          // Check if edge already exists
          const edgeId = relationship.identity.toString();
          if (!cy.getElementById(edgeId).length) {
            elements.push({
              data: {
                id: edgeId,
                source: source.identity.toString(),
                target: target.identity.toString(),
                relationship: relationship.type,
                properties: relationship.properties
              }
            });
          }
        });
        
        // Add new elements to the graph
        cy.add(elements);
        
        // Apply layout only to the newly added elements
        const layout = cy.elements().layout({
          name: 'cose',
          animate: true,
          refresh: 20,
          fit: false
        });
        
        layout.run();
        
      } catch (error) {
        console.error("Error expanding node:", error);
      } finally {
        await session.close();
        await driver.close();
      }
    }
    
    function hideNode(nodeId) {
      cy.getElementById(nodeId).remove();
      document.getElementById('node-details').style.display = 'none';
    }
    
    buildGraphFromNeo4j();
  </script>
</body>
</html>"""

components.html(html, height=800)