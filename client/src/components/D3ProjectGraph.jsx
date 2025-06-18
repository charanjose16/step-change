import React, { useRef, useEffect } from "react";
import * as d3 from "d3";

// Lightweight D3 force-graph for project dependencies
// Props: { graphData: { nodes, edges }, loading, error, fullScreen }
export default function D3ProjectGraph({ graphData, loading, error,  }) {
    const svgRef = useRef();

    useEffect(() => {
        if (!graphData || loading || error) return;
        const container = svgRef.current.parentNode;
        const w = container.offsetWidth > 0 ? container.offsetWidth : 1200;
        const h = container.offsetHeight > 0 ? container.offsetHeight : 900;
        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        // Tooltip div
        let tooltip = d3.select(container).select('.d3-tooltip');
        if (tooltip.empty()) {
            tooltip = d3.select(container).append('div')
                .attr('class', 'd3-tooltip')
                .style('position', 'fixed')
                .style('z-index', 100)
                .style('pointer-events', 'none')
                .style('background', 'rgba(255,255,255,0.97)')
                .style('border', '1px solid #ddd')
                .style('border-radius', '8px')
                .style('padding', '10px 14px')
                .style('font-size', '13px')
                .style('color', '#222')
                .style('box-shadow', '0 2px 8px #0002')
                .style('display', 'none');
        }

        // Pan & zoom support
        const zoom = d3.zoom()
            .scaleExtent([0.2, 3])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
        svg.call(zoom);

        // Main group for pan/zoom
        const g = svg.append('g');

        // Color by type
        const nodeColor = d => {
            if (d.type === 'folder') return '#fbbf24'; // yellow
            if (d.type === 'file') return '#3182bd'; // blue
            if (d.type === 'function') return '#31a354'; // green
            if (d.type === 'important_function') return '#e6550d'; // orange/red
            return '#aaa';
        };
        const edgeColor = d => {
            if (d.type === 'hierarchy') return '#888';
            if (d.type === 'import') return '#3182bd';
            if (d.type === 'call') return '#e6550d';
            return '#aaa';
        };

        // Simulation setup (single, unified graph)
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.edges).id(d => d.id).distance(120))
            .force("charge", d3.forceManyBody().strength(-350))
            .force("center", d3.forceCenter(w / 2, h / 2));

        // Draw edges
        const link = g.append("g")
            .attr("stroke-width", 2)
            .selectAll("line")
            .data(graphData.edges)
            .enter().append("line")
            .attr("stroke", edgeColor)
            .attr("opacity", d => d.type === 'hierarchy' ? 0.6 : 1);

        // Draw nodes
        const node = g.append("g")
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("r", d => d.type === 'file' ? 18 : d.type === 'folder' ? 20 : d.type === 'important_function' ? 14 : 12)
            .attr("fill", nodeColor)
            .on('mouseover', function(event, d) {
                tooltip.style('display', 'block');
                tooltip.html(`<b>${d.type.replace('_', ' ').toUpperCase()}</b><br/>Name: ${d.name || d.id}<br/>` +
                    (d.parent ? `Parent: ${d.parent}<br/>` : '') +
                    (d.id && (d.id.includes('@server') || d.id.includes('@client')) ?
                        `Label: <span style='color:#2563eb'>${d.id.includes('@server') ? '@server' : ''} ${d.id.includes('@client') ? '@client' : ''}</span><br/>` : '')
                );
            })
            .on('mousemove', function(event) {
                tooltip
                    .style('left', (event.clientX + 18) + 'px')
                    .style('top', (event.clientY - 18) + 'px');
            })
            .on('mouseleave', function() {
                tooltip.style('display', 'none');
            })
            .on('click', function(event, d) {
                // Center and zoom to node
                const transform = d3.zoomIdentity
                    .translate(w/2 - d.x * 1.5, h/2 - d.y * 1.5)
                    .scale(1.5);
                svg.transition().duration(500).call(zoom.transform, transform);

                // Highlight logic
                node.classed('highlighted', n => n.id === d.id);
                link.classed('highlighted', l => l.source.id === d.id || l.target.id === d.id);

                // Highlight direct dependencies (edges and nodes)
                const depNodeIds = new Set();
                graphData.edges.forEach(edge => {
                  if (edge.source === d.id) depNodeIds.add(edge.target);
                  if (edge.target === d.id) depNodeIds.add(edge.source);
                });
                node.classed('dep-highlight', n => depNodeIds.has(n.id));
                link.classed('dep-highlight', l => l.source.id === d.id || l.target.id === d.id);

                // Highlight path to root (hierarchy)
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

                // Remove highlight on click outside
                svg.on('click.highlight', function(e) {
                  if (e.target.tagName === 'svg') {
                    node.classed('highlighted', false).classed('dep-highlight', false).classed('root-path', false);
                    link.classed('highlighted', false).classed('dep-highlight', false).classed('root-path', false);
                    svg.on('click.highlight', null);
                  }
                });
            })
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Labels
        g.append("g")
            .selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .attr("font-size", d => d.type === 'file' || d.type === 'folder' ? 14 : 11)
            .attr("fill", "#222")
            .attr("dy", 4)
            .attr("text-anchor", "middle")
            .text(d => {
                if (d.type === 'file' || d.type === 'folder') return d.name || d.id.split('/').pop();
                return d.name || d.id.split(':').pop();
            });

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            g.selectAll("text")
                .attr("x", d => d.x)
                .attr("y", d => d.y - (d.type === 'file' || d.type === 'folder' ? 24 : 16));
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
        // Cleanup
        return () => simulation.stop();
    }, [graphData, loading, error]);

    if (loading) {
        return <div className="flex flex-col items-center justify-center h-96"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>Loading project graph...</div>;
    }
    if (error) {
        return <div className="text-red-600 text-center p-6">{error}</div>;
    }
    if (!graphData) {
        return <div className="text-gray-500 text-center p-6">No graph data to display.</div>;
    }
    return (
        <div className="flex flex-col items-center w-full h-full">
            <div className="relative w-full h-full min-h-[500px] min-w-[500px]" style={{ flex: 1 }}>
                <svg
                    ref={svgRef}
                    style={{ width: '100%', height: '100%', background: '#f8fafc', borderRadius: 12, boxShadow: '0 2px 8px #0001', display: 'block', minHeight: 500, minWidth: 500 }}
                    viewBox={`0 0 1200 900`}
                />
                {/* Legend overlay */}
                <div className="absolute top-4 left-4 bg-white/90 rounded-xl shadow p-3 text-xs flex flex-col gap-1 border border-gray-200 z-10">
                    <div><span className="inline-block w-3 h-3 rounded-full mr-2" style={{background:'#3182bd'}}></span>File (@server/@client)</div>
                    <div><span className="inline-block w-3 h-3 rounded-full mr-2" style={{background:'#31a354'}}></span>Function</div>
                    <div><span className="inline-block w-3 h-3 rounded-full mr-2" style={{background:'#e6550d'}}></span>Important Function</div>
                    <div><span className="inline-block w-3 h-3 rounded-full mr-2" style={{background:'#fbbf24'}}></span>Folder</div>
                    <div><span className="inline-block w-3 h-0.5 align-middle mr-2" style={{background:'#3182bd',width:16}}></span>Import Dependency</div>
                    <div><span className="inline-block w-3 h-0.5 align-middle mr-2" style={{background:'#e6550d',width:16}}></span>Function Call</div>
                    <div><span className="inline-block w-3 h-0.5 align-middle mr-2" style={{background:'#888',width:16}}></span>Hierarchy Edge</div>
                </div>
            </div>
        </div>
    );
}
