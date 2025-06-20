import React, { useRef, useEffect } from "react";
import * as d3 from "d3";
import { useState } from "react";

// Lightweight D3 force-graph for project dependencies
// Props: { graphData: { nodes, edges }, loading, error }
export default function D3ProjectGraph({ graphData, loading, error }) {
    const svgRef = useRef();

    useEffect(() => {
        if (!graphData || loading || error) return;
        const container = svgRef.current.parentNode;
        const w = container.offsetWidth > 0 ? container.offsetWidth : 1400;
        const h = container.offsetHeight > 0 ? container.offsetHeight : 1000;
        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        // Background grid
        svg.append("defs")
            .append("pattern")
            .attr("id", "grid")
            .attr("width", 20)
            .attr("height", 20)
            .append("path")
            .attr("d", "M 20 0 L 0 0 0 20")
            .attr("stroke", "#e5e7eb")
            .attr("stroke-width", 0.5);

        svg.append("rect")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("fill", "url(#grid)");

        // Tooltip
        let tooltip = d3.select(container).select('.d3-tooltip');
        if (tooltip.empty()) {
            tooltip = d3.select(container).append('div')
                .attr('class', 'd3-tooltip')
                .style('position', 'fixed')
                .style('z-index', 200)
                .style('pointer-events', 'none')
                .style('background', 'linear-gradient(145deg, #ffffff, #f1f5f9)')
                .style('border', '1px solid #d1d5db')
                .style('border-radius', '10px')
                .style('padding', '12px 16px')
                .style('font-size', '14px')
                .style('color', '#1f2937')
                .style('box-shadow', '0 6px 20px rgba(0,0,0,0.15)')
                .style('display', 'none')
                .style('max-width', '300px');
        }

        // Context menu
        let contextMenu = d3.select(container).select('.d3-context-menu');
        if (contextMenu.empty()) {
            contextMenu = d3.select(container).append('div')
                .attr('class', 'd3-context-menu')
                .style('position', 'fixed')
                .style('z-index', 201)
                .style('background', '#ffffff')
                .style('border', '1px solid #d1d5db')
                .style('border-radius', '8px')
                .style('padding', '8px 0')
                .style('box-shadow', '0 4px 16px rgba(0,0,0,0.2)')
                .style('display', 'none');
        }

        // Pan & zoom
        const zoom = d3.zoom()
            .scaleExtent([0.3, 5])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
        svg.call(zoom);

        const g = svg.append('g');

        // Colors
        const nodeColor = d => {
            if (d.type === 'folder') return 'url(#folderGradient)';
            if (d.type === 'file') return 'url(#fileGradient)';
            if (d.type === 'function') return 'url(#functionGradient)';
            if (d.type === 'important_function') return 'url(#importantFunctionGradient)';
            return '#9ca3af';
        };
        const edgeColor = d => {
            if (d.type === 'hierarchy') return '#9ca3af';
            if (d.type === 'import') return '#3b82f6';
            if (d.type === 'call') return '#ef4444';
            return '#d1d5db';
        };

        // Gradients
        const defs = svg.append("defs");
        defs.append("linearGradient")
            .attr("id", "folderGradient")
            .attr("x1", "0%").attr("y1", "0%").attr("x2", "100%").attr("y2", "100%")
            .call(g => g.append("stop").attr("offset", "0%").attr("stop-color", "#f59e0b"))
            .call(g => g.append("stop").attr("offset", "100%").attr("stop-color", "#d97706"));
        defs.append("linearGradient")
            .attr("id", "fileGradient")
            .attr("x1", "0%").attr("y1", "0%").attr("x2", "100%").attr("y2", "100%")
            .call(g => g.append("stop").attr("offset", "0%").attr("stop-color", "#3b82f6"))
            .call(g => g.append("stop").attr("offset", "100%").attr("stop-color", "#2563eb"));
        defs.append("linearGradient")
            .attr("id", "functionGradient")
            .attr("x1", "0%").attr("y1", "0%").attr("x2", "100%").attr("y2", "100%")
            .call(g => g.append("stop").attr("offset", "0%").attr("stop-color", "#22c55e"))
            .call(g => g.append("stop").attr("offset", "100%").attr("stop-color", "#16a34a"));
        defs.append("linearGradient")
            .attr("id", "importantFunctionGradient")
            .attr("x1", "0%").attr("y1", "0%").attr("x2", "100%").attr("y2", "100%")
            .call(g => g.append("stop").attr("offset", "0%").attr("stop-color", "#ef4444"))
            .call(g => g.append("stop").attr("offset", "100%").attr("stop-color", "#dc2626"));

        // Simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.edges).id(d => d.id).distance(200))
            .force("charge", d3.forceManyBody().strength(-800))
            .force("center", d3.forceCenter(w / 2, h / 2))
            .force("collide", d3.forceCollide().radius(d => d.type === 'folder' ? 35 : 25));

        // Edges
        const link = g.append("g")
            .attr("stroke-width", 2.5)
            .selectAll("path")
            .data(graphData.edges)
            .enter().append("path")
            .attr("stroke", edgeColor)
            .attr("fill", "none")
            .attr("opacity", d => d.type === 'hierarchy' ? 0.4 : 0.8)
            .attr("stroke-dasharray", d => d.type === 'import' ? "6,6" : null);

        // Nodes
        const node = g.append("g")
            .attr("stroke", "#fff")
            .attr("stroke-width", 2)
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("r", d => d.type === 'folder' ? 26 : d.type === 'file' ? 22 : d.type === 'important_function' ? 18 : 16)
            .attr("fill", nodeColor)
            .style("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.2))")
            .on('mouseover', function(event, d) {
                d3.select(this).transition().duration(200).attr("r", d => d.type === 'folder' ? 30 : d.type === 'file' ? 26 : d.type === 'important_function' ? 22 : 20);
                tooltip.style('display', 'block').html(`
                    <div class="font-bold">${d.type.replace('_', ' ').toUpperCase()}</div>
                    <div>Name: ${d.name || d.id}</div>
                    ${d.parent ? `<div>Parent: ${d.parent}</div>` : ''}
                    ${d.id && (d.id.includes('@server') || d.id.includes('@client')) ?
                        `<div>Label: <span class="text-blue-600">${d.id.includes('@server') ? '@server' : ''} ${d.id.includes('@client') ? '@client' : ''}</span></div>` : ''}
                `);
                // Glow connected edges
                link.transition().duration(200)
                    .attr("stroke-width", l => l.source.id === d.id || l.target.id === d.id ? 4 : 2.5)
                    .attr("opacity", l => l.source.id === d.id || l.target.id === d.id ? 1 : d.type === 'hierarchy' ? 0.4 : 0.8);
                // Highlight related nodes
                const relatedNodes = new Set();
                graphData.edges.forEach(edge => {
                    if (edge.source.id === d.id) relatedNodes.add(edge.target.id);
                    if (edge.target.id === d.id) relatedNodes.add(edge.source.id);
                });
                node.transition().duration(200)
                    .attr("opacity", n => n.id === d.id || relatedNodes.has(n.id) ? 1 : 0.5);
            })
            .on('mousemove', function(event) {
                tooltip
                    .style('left', (event.clientX + 25) + 'px')
                    .style('top', (event.clientY - 25) + 'px');
            })
            .on('mouseleave', function() {
                d3.select(this).transition().duration(200).attr("r", d => d.type === 'folder' ? 26 : d.type === 'file' ? 22 : d.type === 'important_function' ? 18 : 16);
                tooltip.style('display', 'none');
                link.transition().duration(200)
                    .attr("stroke-width", 2.5)
                    .attr("opacity", l => l.type === 'hierarchy' ? 0.4 : 0.8);
                node.transition().duration(200).attr("opacity", 1);
            })
            .on('contextmenu', function(event, d) {
                event.preventDefault();
                contextMenu.style('display', 'block')
                    .style('left', (event.clientX + 10) + 'px')
                    .style('top', (event.clientY + 10) + 'px')
                    .html(`
                        <div class="px-4 py-2 hover:bg-gray-100 cursor-pointer" onclick="window.dispatchEvent(new CustomEvent('zoomToNode', { detail: '${d.id}' }))">Zoom to Node</div>
                        <div class="px-4 py-2 hover:bg-gray-100 cursor-pointer" onclick="window.dispatchEvent(new CustomEvent('highlightChain', { detail: '${d.id}' }))">Highlight Dependency Chain</div>
                        <div class="px-4 py-2 hover:bg-gray-100 cursor-pointer" onclick="window.dispatchEvent(new CustomEvent('startPathExplorer', { detail: '${d.id}' }))">Start Dependency Path Explorer</div>
                    `);
            })
            .on('click', function(event, d) {
                const transform = d3.zoomIdentity
                    .translate(w/2 - d.x * 2.5, h/2 - d.y * 2.5)
                    .scale(2.5);
                svg.transition().duration(750).call(zoom.transform, transform);
                node.classed('highlighted dep-highlight root-path chain-highlight path-explorer', false);
                link.classed('highlighted dep-highlight root-path chain-highlight path-explorer', false);
                node.classed('highlighted', n => n.id === d.id);
                link.classed('highlighted', l => l.source.id === d.id || l.target.id === d.id);
                const depNodeIds = new Set();
                graphData.edges.forEach(edge => {
                    if (edge.source.id === d.id) depNodeIds.add(edge.target.id);
                    if (edge.target.id === d.id) depNodeIds.add(edge.source.id);
                });
                node.classed('dep-highlight', n => depNodeIds.has(n.id));
                link.classed('dep-highlight', l => l.source.id === d.id || l.target.id === d.id);
                let current = d;
                const pathToRoot = new Set();
                while (current && current.parent) {
                    pathToRoot.add(current.id);
                    const parentNode = graphData.nodes.find(n => n.id === current.parent);
                    if (!parentNode) break;
                    pathToRoot.add(parentNode.id);
                    current = parentNode;
                }
                node.classed('root-path', n => pathToRoot.has(n.id));
                link.classed('root-path', l => pathToRoot.has(l.source.id) && pathToRoot.has(l.target.id) && l.type === 'hierarchy');
            })
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Labels
        const label = g.append("g")
            .selectAll("g")
            .data(graphData.nodes)
            .enter().append("g");
        
        label.append("rect")
            .attr("fill", "#ffffff")
            .attr("opacity", 0.9)
            .attr("rx", 6)
            .attr("ry", 6)
            .attr("x", d => d.x - Math.min((d.name || d.id.split('/').pop()).length * 4.5, 100))
            .attr("y", d => d.y - (d.type === 'file' || d.type === 'folder' ? 36 : 28))
            .attr("width", d => Math.min((d.name || d.id.split('/').pop()).length * 9, 200))
            .attr("height", 24)
            .style("filter", "drop-shadow(0 1px 2px rgba(0,0,0,0.1))");

        label.append("text")
            .attr("font-size", d => d.type === 'file' || d.type === 'folder' ? 16 : 13)
            .attr("fill", "#1f2937")
            .attr("dy", 4)
            .attr("text-anchor", "middle")
            .text(d => {
                const text = d.name || d.id.split('/').pop();
                return text.length > 20 ? text.substring(0, 17) + '...' : text;
            });

        // Dependency Path Explorer
        let pathStartNode = null;
        window.addEventListener('startPathExplorer', (e) => {
            pathStartNode = e.detail;
            tooltip.style('display', 'block').html('Right-click another node to find the shortest dependency path.');
        });

        window.addEventListener('zoomToNode', (e) => {
            const d = graphData.nodes.find(n => n.id === e.detail);
            if (d) {
                const transform = d3.zoomIdentity.translate(w/2 - d.x * 2.5, h/2 - d.y * 2.5).scale(2.5);
                svg.transition().duration(750).call(zoom.transform, transform);
            }
            contextMenu.style('display', 'none');
        });

        window.addEventListener('highlightChain', (e) => {
            const chainNodeIds = new Set([e.detail]);
            const chainEdgeIds = new Set();
            const queue = [e.detail];
            while (queue.length > 0) {
                const currentId = queue.shift();
                graphData.edges.forEach(edge => {
                    if (edge.type === 'import' || edge.type === 'call') {
                        if (edge.source.id === currentId && !chainNodeIds.has(edge.target.id)) {
                            chainNodeIds.add(edge.target.id);
                            chainEdgeIds.add(JSON.stringify(edge));
                            queue.push(edge.target.id);
                        } else if (edge.target.id === currentId && !chainNodeIds.has(edge.source.id)) {
                            chainNodeIds.add(edge.source.id);
                            chainEdgeIds.add(JSON.stringify(edge));
                            queue.push(edge.source.id);
                        }
                    }
                });
            }
            node.classed('chain-highlight', n => chainNodeIds.has(n.id));
            link.classed('chain-highlight', l => chainEdgeIds.has(JSON.stringify(l)));
            contextMenu.style('display', 'none');
        });

        window.addEventListener('contextmenu', (e) => {
            if (e.target.tagName !== 'circle') {
                contextMenu.style('display', 'none');
                if (pathStartNode) {
                    tooltip.style('display', 'none');
                    pathStartNode = null;
                    node.classed('path-explorer', false);
                    link.classed('path-explorer', false);
                }
            }
        });

        simulation.on("tick", () => {
            link.attr("d", d => {
                return `M${d.source.x},${d.source.y} L${d.target.x},${d.target.y}`;
            });

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            label.selectAll("rect")
                .attr("x", d => d.x - Math.min((d.name || d.id.split('/').pop()).length * 4.5, 100))
                .attr("y", d => d.y - (d.type === 'file' || d.type === 'folder' ? 36 : 28));

            label.selectAll("text")
                .attr("x", d => d.x)
                .attr("y", d => d.y - (d.type === 'file' || d.type === 'folder' ? 26 : 18));
        });

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        return () => {
            simulation.stop();
            window.removeEventListener('zoomToNode', () => {});
            window.removeEventListener('highlightChain', () => {});
            window.removeEventListener('startPathExplorer', () => {});
            window.removeEventListener('contextmenu', () => {});
        };
    }, [graphData, loading, error]);

    const [legendOpen, setLegendOpen] = useState(true);

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center h-full">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                <span className="text-gray-700 font-medium">Loading project graph...</span>
            </div>
        );
    }
    if (error) {
        return <div className="text-red-600 text-center p-8 font-medium">{error}</div>;
    }
    if (!graphData) {
        return <div className="text-gray-600 text-center p-8 font-medium">No graph data to display.</div>;
    }

    return (
        <div className="flex flex-col items-center w-full h-full">
            <style>{`
                .d3-tooltip { transition: all 0.2s ease; }
                .d3-context-menu div:hover { background-color: #f3f4f6; }
                .highlighted { stroke: #000 !important; stroke-width: 3 !important; }
                .dep-highlight { opacity: 0.8 !important; }
                .root-path { stroke: #6b7280 !important; stroke-width: 2.5 !important; }
                .chain-highlight { stroke: #eab308 !important; stroke-width: 4 !important; opacity: 1 !important; }
                .path-explorer { stroke: #8b5cf6 !important; stroke-width: 4 !important; opacity: 1 !important; }
                .legend-toggle-btn {
                    position: absolute;
                    top: 1rem;
                    left: 1rem;
                    background: #fff;
                    border-radius: 50%;
                    width: 32px;
                    height: 32px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 2px 8px #0001;
                    border: 1px solid #e5e7eb;
                    cursor: pointer;
                    z-index: 20;
                    transition: background 0.15s;
                }
                .legend-toggle-btn:hover { background: #f3f4f6; }
            `}</style>
            <div className="relative w-full h-full min-h-[400px] min-w-[400px] flex-1" style={{ flex: 1 }}>
                <svg
                    ref={svgRef}
                    style={{ width: '100%', height: '100%', background: '#fff', borderRadius: 16, boxShadow: '0 6px 20px rgba(0,0,0,0.08)', display: 'block' }}
                    viewBox={`100 100 1400 1000`}
                />
                {/* Toggle Button */}
                {!legendOpen && (
                    <button
                        className="legend-toggle-btn"
                        aria-label="Show legend"
                        onClick={() => setLegendOpen(true)}
                        style={{ zIndex: 20 }}
                        
                    >
                        <svg width="18" height="18" fill="none" viewBox="0 0 20 20"><rect x="3" y="7" width="14" height="6" rx="2" fill="#6366f1"/><rect x="7" y="3" width="6" height="14" rx="2" fill="#6366f1"/></svg>
                    </button>
                )}
                {/* Legend Panel */}
                {legendOpen && (
                    <div className="absolute top-4 left-4 bg-white/95 rounded-lg shadow-lg px-2 py-1 text-xs flex flex-col gap-0.5 border border-gray-200 z-10 min-w-[140px] max-w-[180px]"
                        style={{ fontSize: '11px', lineHeight: 1.2, boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
                        <div className="mb-0.5 flex justify-between items-center">
                            <span className="font-semibold text-gray-700 text-xs">Legend</span>
                            
                        </div>
                        <div className="flex items-center"><span className="inline-block w-2.5 h-2.5 rounded-full mr-1.5 bg-gradient-to-br from-amber-500 to-amber-600"></span>File (@server/@client)</div>
                        <div className="flex items-center"><span className="inline-block w-2.5 h-2.5 rounded-full mr-1.5 bg-gradient-to-br from-green-500 to-green-600"></span>Function</div>
                        <div className="flex items-center"><span className="inline-block w-2.5 h-2.5 rounded-full mr-1.5 bg-gradient-to-br from-red-500 to-red-600"></span>Important Function</div>
                        <div className="flex items-center"><span className="inline-block w-2.5 h-2.5 rounded-full mr-1.5 bg-gradient-to-br from-yellow-500 to-yellow-600"></span>Folder</div>
                        <div className="flex items-center"><span className="inline-block w-3 h-0.5 mr-1.5 bg-blue-600"></span>Import Dependency</div>
                        <div className="flex items-center"><span className="inline-block w-3 h-0.5 mr-1.5 bg-red-600"></span>Function Call</div>
                        <div className="flex items-center"><span className="inline-block w-3 h-0.5 mr-1.5 bg-gray-400"></span>Hierarchy Edge</div>
                        
                    </div>
                )}
            </div>
        </div>
    );
}