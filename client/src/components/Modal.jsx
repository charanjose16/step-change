import React, { useEffect } from "react";
import { X } from "lucide-react";

export default function Modal({ isOpen, onClose, children, className = "" }) {
    useEffect(() => {
        document.body.style.overflow = isOpen ? "hidden" : "auto";
        return () => {
            document.body.style.overflow = "auto";
        };
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
            <div
                className="fixed inset-0 bg-black opacity-50"
                onClick={onClose}
            ></div>
            {/* Modal container */}
            <div
                className={`relative bg-white rounded-xl w-full max-w-6xl max-h-full flex flex-col shadow-lg ${className}`}
            >
                <button
                    onClick={onClose}
                    className="absolute top-2 right-2 text-teal-600 hover:text-teal-800 hover:bg-teal-100 p-1 rounded-full z-10 transition-colors shadow-lg"
                >
                    <X className="h-6 w-6" />
                </button>
                <div className="flex-1 min-h-0">
                    {children}
                </div>
            </div>
        </div>
    );
}