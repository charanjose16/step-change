import React, { useEffect } from "react";
import { X } from "lucide-react";

export default function Modal({ isOpen, onClose, children }) {
    useEffect(() => {
        document.body.style.overflow = isOpen ? "hidden" : "auto";
        return () => {
            document.body.style.overflow = "auto";
        };
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 flex items-center justify-center z-50 p-4"> {/* Added padding here for smaller screens */}
            <div
                className="fixed inset-0 bg-black opacity-50"
                onClick={onClose}
            ></div>
            {/* Modal container - Let children control height and scroll */}
            <div className="relative bg-white rounded w-full max-w-6xl max-h-full flex flex-col"> {/* Added flex flex-col */}
                <button
                    onClick={onClose}
                    className="absolute top-2 right-2 text-gray-600 hover:text-gray-900 z-10" /* Ensure button is above content */
                >
                    <X className="h-6 w-6" />
                </button>
                {/* Removed padding and overflow styles from here */}
                <div className="flex-1 min-h-0"> {/* Allow this div to shrink and grow; min-h-0 prevents flexbox overflow issues */}
                    {children}
                </div>
            </div>
        </div>
    );
}
