import React from 'react';
import MermaidGraph from './MermaidGraph';
import { ArrowLeft, Maximize, Loader2, AlertTriangle, ChevronDown } from 'lucide-react';

export default function RequirementDetailView({
    requirement,
    graphResponses,
    isLoadingGraph,
    graphError,
    selectedGraphIndex,
    onGraphIndexChange,
    onGoBack,
    onFullscreen,
}) {
    const hasGraphs = graphResponses.length > 0;
    const selectedGraphData = hasGraphs ? graphResponses[selectedGraphIndex] : null;
    const chartCode = selectedGraphData?.generated_code;

    // SVG for custom dropdown arrow (copied from original)
    const customArrow = `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e")`;

    return (
        <div className="flex flex-col">
            {/* Header: Back Button & Title */}
            <div className="flex items-center mb-4 flex-shrink-0">
                <button
                    onClick={onGoBack}
                    className="p-2 rounded-full text-slate-600 hover:bg-slate-100 hover:text-slate-800 transition-colors mr-2"
                    title="Back to List"
                >
                    <ArrowLeft className="h-5 w-5" />
                </button>
                <h3
                    className="text-lg font-semibold text-slate-800 truncate"
                    title={requirement?.file_name || "File Details"}
                >
                    {requirement?.file_name || "File Details"}
                </h3>
            </div>

            {/* Content Area: Requirements and Graph */}
            <div className="border border-slate-200 rounded-lg bg-white overflow-hidden">
                {/* Requirement Text */}
                <div className="p-4 border-b border-slate-200 bg-slate-50">
                    <p
                        className="text-xs text-slate-500 mb-2 truncate"
                        title={requirement?.relative_path || "No path available"}
                    >
                        Path: {requirement?.relative_path || "N/A"}
                    </p>
                    <p className="text-sm text-slate-800 whitespace-pre-wrap break-words">
                        {requirement?.requirements || "No requirements description."}
                    </p>
                </div>

                {/* Graph Section */}
                <div className="p-4">
                    {/* Graph Selector & Fullscreen Button */}
                    <div className="flex justify-between items-center mb-3 gap-4">
                        <label
                            htmlFor="graph-select"
                            className="text-sm font-medium text-slate-700 flex-shrink-0"
                        >
                            Select Graph:
                        </label>
                        {hasGraphs ? (
                            <select
                                id="graph-select"
                                value={selectedGraphIndex}
                                onChange={(e) => onGraphIndexChange(Number(e.target.value))}
                                className="block w-full pl-3 pr-8 py-2 border border-slate-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white appearance-none"
                                style={{
                                    backgroundImage: customArrow,
                                    backgroundPosition: 'right 0.5rem center',
                                    backgroundRepeat: 'no-repeat',
                                    backgroundSize: '1.25em 1.25em',
                                }}
                                disabled={graphResponses.length <= 1}
                            >
                                {graphResponses.map((graph, i) => (
                                    <option key={i} value={i}>
                                        {graph.target_graph || `Graph ${i + 1}`}
                                    </option>
                                ))}
                            </select>
                        ) : (
                            <span className="text-sm text-slate-500 italic w-full text-right">
                                {isLoadingGraph
                                    ? "Loading..."
                                    : graphError || "No graphs available"}
                            </span>
                        )}
                        {hasGraphs && chartCode && (
                            <button
                                onClick={onFullscreen}
                                className="p-1.5 text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 rounded flex-shrink-0"
                                title="Fullscreen Graph"
                            >
                                <Maximize className="h-5 w-5" />
                            </button>
                        )}
                    </div>

                    {/* Graph Display Area */}
                    <div className="bg-gray-50 p-2 rounded border border-slate-200 min-h-[300px] flex items-center justify-center relative">
                        {isLoadingGraph ? (
                            <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
                                <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
                            </div>
                        ) : graphError && !hasGraphs ? (
                            <div className="flex flex-col items-center justify-center text-red-600 text-center p-4">
                                <AlertTriangle className="h-8 w-8 mb-2" />
                                <span className="text-sm">{graphError}</span>
                            </div>
                        ) : hasGraphs && chartCode ? (
                            <MermaidGraph chart={chartCode} />
                        ) : hasGraphs && !chartCode ? (
                            <div className="flex items-center justify-center h-full text-slate-500 p-4 text-center">
                                <AlertTriangle className="w-5 h-5 mr-2 flex-shrink-0" />{" "}
                                Selected graph data is empty or invalid.
                            </div>
                        ) : (
                            <div className="flex items-center justify-center h-full text-slate-500 p-4 text-center">
                                No graphs available for this file's requirements.
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

