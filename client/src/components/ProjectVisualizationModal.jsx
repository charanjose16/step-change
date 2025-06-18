import React, { useState } from "react";
import FullScreenGraph from "./FullScreenGraph";
import D3ProjectGraph from "./D3ProjectGraph";

export default function ProjectVisualizationModal({
    show,
    onClose,
    folderName,
}) {
    const [projectGraphData, setProjectGraphData] = useState(null);
    const [loadingProjectGraph, setLoadingProjectGraph] = useState(false);
    const [projectGraphError, setProjectGraphError] = useState("");

    React.useEffect(() => {
        if (!show) return;
        async function fetchGraph() {
            setLoadingProjectGraph(true);
            setProjectGraphData(null);
            setProjectGraphError("");
            try {
                let cleanFolderName = folderName;
                if (cleanFolderName && cleanFolderName.includes("/")) {
                    cleanFolderName = cleanFolderName.split("/").slice(-2).join("/");
                }
                // Validate folder name before making the request
                const validFolderName = cleanFolderName && /^[\w\-/]+\/?$/.test(cleanFolderName);
                if (!validFolderName) {
                    setProjectGraphError("Invalid folder name. Use only letters, numbers, slashes, hyphens, or underscores.");
                    setLoadingProjectGraph(false);
                    return;
                }
                const response = await import("../api/axiosConfig").then(({ default: axios }) =>
                    axios.post("/analysis/project-graph", { folder_name: cleanFolderName })
                );
                if (response.data && response.data.success && response.data.graph) {
                    setProjectGraphData(response.data.graph);
                } else {
                    setProjectGraphError("No graph data returned from backend.");
                }
            } catch (err) {
                // Handle FastAPI validation errors (422)
                if (err?.response?.status === 422 && err?.response?.data?.detail) {
                    setProjectGraphError(
                        Array.isArray(err.response.data.detail)
                        ? err.response.data.detail.map(e => e.msg).join('; ')
                        : (typeof err.response.data.detail === 'string' ? err.response.data.detail : 'Invalid folder name or request.')
                    );
                } else if (typeof err?.response?.data?.detail === 'object') {
                    setProjectGraphError('Backend error: ' + JSON.stringify(err.response.data.detail));
                } else {
                    setProjectGraphError(err?.response?.data?.detail || err.message || "Failed to fetch project graph.");
                }
            } finally {
                setLoadingProjectGraph(false);
            }
        }
        fetchGraph();
    }, [show, folderName]);

    if (!show) return null;
    return show ? (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-62">
        <div className={`relative bg-white rounded-lg shadow-2xl p-2 w-[98vw] h-[90vh] max-w-[98vw] max-h-[98vh] flex flex-col items-stretch`}>
            <button
                className="absolute top-2 right-2 text-gray-600 hover:text-red-500 text-xl z-20"
                onClick={() => {
                    setProjectGraphData(null);
                    setProjectGraphError("");
                    onClose();
                }}
                aria-label="Close"
            >
                &times;
            </button>
            <div className="absolute top-2 left-2 z-20">
                {/* Fullscreen button toggles a fullscreen state (optional, here always maximized) */}
            </div>
            <div className="flex-1 flex items-center justify-center min-h-0 min-w-0">
                <D3ProjectGraph
                    graphData={projectGraphData}
                    loading={loadingProjectGraph}
                    error={projectGraphError}
                    fullScreen={true}
                />
            </div>
        </div>
    </div>
) : null;
}
