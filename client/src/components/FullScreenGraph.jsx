import React, { useState, useEffect, useRef } from "react";
import MermaidGraph from "./MermaidGraph";
import { X } from "lucide-react";

function FullScreenGraph({ chart, onClose }) {
    const initialZoomLevel = 1.5; // Set initial zoom level (e.g., 150%)
    const [zoom, setZoom] = useState(initialZoomLevel);
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStart = useRef({ x: 0, y: 0 });
    const offsetStart = useRef({ x: 0, y: 0 });
    const containerRef = useRef(null);
    const contentRef = useRef(null); // Ref for the content being transformed

    // Effect to handle body scroll lock (global wheel listener removed)
    useEffect(() => {
        // Prevent background scrolling when the modal is open
        document.body.style.overflow = 'hidden';

        // Attempt to focus the container
        containerRef.current?.focus();

        return () => {
            document.body.style.overflow = 'auto';
        };
    }, []); // Empty dependency array ensures this runs only on mount and unmount

    // Reset zoom and offset when the chart changes or component mounts
    useEffect(() => {
        // Calculate initial offset to roughly center the zoomed content
        let initialOffsetX = 0;
        let initialOffsetY = 0;
        if (containerRef.current) {
            const containerRect = containerRef.current.getBoundingClientRect();
            initialOffsetX = (containerRect.width / 2) * (1 - initialZoomLevel);
            initialOffsetY = (containerRect.height / 2) * (1 - initialZoomLevel);
        }

        setOffset({ x: initialOffsetX, y: initialOffsetY });
        setZoom(initialZoomLevel); // Start at the defined initial zoom level
    }, [chart]); // Recalculate when chart changes

    // Container's specific wheel handler for zoom logic
    const handleWheel = (e) => {
        if (e.ctrlKey) {
            e.preventDefault(); // Prevent browser zoom
            e.stopPropagation();
            console.log('Container wheel handler running zoom logic'); // DEBUG

            if (Math.abs(e.deltaY) < 2) return; // Ignore minor wheel events

            const scaleAmount = 1.1;
            const point = { x: e.clientX, y: e.clientY };
            const rect = containerRef.current.getBoundingClientRect();
            const mouseX = point.x - rect.left;
            const mouseY = point.y - rect.top;
            const mousePointToContentX = (mouseX - offset.x) / zoom;
            const mousePointToContentY = (mouseY - offset.y) / zoom;

            let newZoom = zoom;
            if (e.deltaY < 0) {
                newZoom = Math.min(zoom * scaleAmount, 10);
            } else {
                newZoom = Math.max(zoom / scaleAmount, 0.1);
            }

            const newOffsetX = mouseX - mousePointToContentX * newZoom;
            const newOffsetY = mouseY - mousePointToContentY * newZoom;

            setZoom(newZoom);
            setOffset({ x: newOffsetX, y: newOffsetY });
        }
        // No preventDefault needed here if not using ctrlKey
    };

    // --- Mouse Drag Handlers (handleMouseDown, handleMouseMove, handleMouseUpOrLeave) ---
    const handleMouseDown = (e) => {
        // Allow dragging with left mouse button
        if (e.button !== 0) return;
        setIsDragging(true);
        dragStart.current = { x: e.clientX, y: e.clientY };
        offsetStart.current = { ...offset };
        if (containerRef.current) containerRef.current.style.cursor = "grabbing";
        e.stopPropagation(); // Prevent triggering onClick on the backdrop
    };

    const handleMouseMove = (e) => {
        if (!isDragging) return;
        const dx = e.clientX - dragStart.current.x;
        const dy = e.clientY - dragStart.current.y;
        setOffset({ x: offsetStart.current.x + dx, y: offsetStart.current.y + dy });
    };

    const handleMouseUpOrLeave = (e) => {
        if (isDragging) {
            setIsDragging(false);
            if (containerRef.current) containerRef.current.style.cursor = "grab";
            // Check if it was a drag or just a click before stopping propagation
            const dx = Math.abs(e.clientX - dragStart.current.x);
            const dy = Math.abs(e.clientY - dragStart.current.y);
            if (dx > 5 || dy > 5) { // Only stop propagation if it was a drag
                 e.stopPropagation();
            }
        }
    };

    // --- Zoom Button Handlers (zoomIn, zoomOut) ---
    const zoomIn = () => {
        // Zoom towards center
        const scaleAmount = 1.2;
        const newZoom = Math.min(zoom * scaleAmount, 10);
        if (containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect();
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            // Calculate point relative to content's current top-left
            const centerPointToContentX = (centerX - offset.x) / zoom;
            const centerPointToContentY = (centerY - offset.y) / zoom;
            // Calculate new offset to keep the center point centered
            const newOffsetX = centerX - centerPointToContentX * newZoom;
            const newOffsetY = centerY - centerPointToContentY * newZoom;
            setZoom(newZoom);
            setOffset({ x: newOffsetX, y: newOffsetY });
        }
    };

    const zoomOut = () => {
        // Zoom out from center
        const scaleAmount = 1.2;
        const newZoom = Math.max(zoom / scaleAmount, 0.1);
         if (containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect();
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            // Calculate point relative to content's current top-left
            const centerPointToContentX = (centerX - offset.x) / zoom;
            const centerPointToContentY = (centerY - offset.y) / zoom;
            // Calculate new offset to keep the center point centered
            const newOffsetX = centerX - centerPointToContentX * newZoom;
            const newOffsetY = centerY - centerPointToContentY * newZoom;
            setZoom(newZoom);
            setOffset({ x: newOffsetX, y: newOffsetY });
        }
    };

    // --- Keyboard Listener (Esc) ---
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose]);

    // --- JSX Rendering ---
    return (
        <div
            className="fixed inset-0 z-[60] bg-black bg-opacity-75 flex items-center justify-center"
            onClick={onClose} // Close on backdrop click
        >
            {/* Close Button */}
            <button
                onClick={(e) => { e.stopPropagation(); onClose(); }}
                className="absolute top-8 right-12 z-[70] text-white bg-black bg-opacity-50 rounded-full p-2 hover:bg-opacity-75 transition-opacity"
                title="Close Fullscreen (Esc)"
            >
                <X className="h-6 w-6" />
            </button>

            {/* Container for Pan/Zoom */}
            <div
                ref={containerRef}
                tabIndex={-1}
                onWheel={handleWheel} // Keep this for zoom logic execution
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUpOrLeave}
                onMouseLeave={handleMouseUpOrLeave}
                onClick={(e) => e.stopPropagation()}
                className="relative w-[95vw] h-[95vh] bg-slate-100 rounded shadow-xl overflow-hidden select-none outline-none"
                style={{ cursor: isDragging ? "grabbing" : "grab" }}
            >
                {/* Overlay Controls */}
                <div className="absolute top-4 left-4 z-10 flex flex-col space-y-2">
                    <button
                        onClick={zoomIn}
                        className="p-2 bg-white rounded shadow hover:bg-gray-200 transition-colors"
                        title="Zoom In"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                    </button>
                    <button
                        onClick={zoomOut}
                        className="p-2 bg-white rounded shadow hover:bg-gray-200 transition-colors"
                        title="Zoom Out"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                        </svg>
                    </button>
                    <div className="text-xs text-gray-500 bg-white bg-opacity-70 px-1 py-0.5 rounded">
                        Ctrl+Scroll to Zoom
                    </div>
                </div>

                {/* Graph Content */}
                <div
                    ref={contentRef}
                    style={{
                        position: 'absolute',
                        top: 0, left: 0, width: '100%', height: '100%',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
                        transformOrigin: "0 0",
                        transition: isDragging ? 'none' : 'transform 0.1s ease-out',
                    }}
                >
                    <MermaidGraph chart={chart} />
                </div>
            </div>
        </div>
    );
}

export default FullScreenGraph;