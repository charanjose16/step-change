import React from 'react';
import { Eye } from 'lucide-react';
import { trimDescription } from '../utils/textUtils';

export default function RequirementListView({
    requirements,
    totalFilteredRequirements,
    searchTerm,
    onSearchChange,
    currentPage,
    totalPages,
    onPageChange,
    onSelectRequirement,
    requirementsPerPage,
    startIdx,
}) {
    // Pagination number generation logic
    const maxVisiblePages = 5;
    let pageNumbers = [];
    if (totalPages <= maxVisiblePages) {
        pageNumbers = Array.from({ length: totalPages }, (_, i) => i + 1);
    } else {
        let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
        let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
        if (endPage - startPage + 1 < maxVisiblePages) {
            startPage = Math.max(1, endPage - maxVisiblePages + 1);
        }
        if (startPage > 1) pageNumbers.push(1, '...');
        for (let i = startPage; i <= endPage; i++) {
            pageNumbers.push(i);
        }
        if (endPage < totalPages) pageNumbers.push('...', totalPages);
    }

    return (
        <div className="flex flex-col">
            {/* Search Input */}
            <div className="mb-4 flex-shrink-0">
                <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => onSearchChange(e.target.value)}
                    placeholder="Filter by file name..."
                    className="w-full px-3 py-2 border border-teal-300/40 rounded-lg shadow-lg bg-white text-teal-800 focus:outline-none focus:ring-2 focus:ring-teal-300 sm:text-sm"
                />
            </div>

            {/* List Area */}
            <div className="mb-4">
                {totalFilteredRequirements === 0 ? (
                    <div className="flex flex-col items-center justify-center text-center py-10">
                        <p className="text-red-400 text-lg">
                            {searchTerm
                                ? "No files match your search."
                                : "No requirements found."}
                        </p>
                    </div>
                ) : (
                    <ul className="space-y-3">
                        {requirements.map((req, idx) => (
                            <li
                                key={startIdx + idx}
                                className="p-4 border border-teal-200 rounded-lg bg-teal-50 hover:bg-teal-100 transition-colors duration-150 cursor-pointer group shadow-lg"
                                onClick={() => onSelectRequirement(req)}
                            >
                                <div className="flex justify-between items-center">
                                    <div className="flex-1 mr-4 overflow-hidden">
                                        <h3
                                            className="text-base font-semibold text-teal-800 mb-1 group-hover:text-teal-600 truncate"
                                            title={req.file_name || "Unknown File"}
                                        >
                                            {req.file_name || "Unknown File"}
                                        </h3>
                                        <p
                                            className="text-xs text-teal-600 mb-2 truncate"
                                            title={req.relative_path || "No path available"}
                                        >
                                            {req.relative_path || "No path available"}
                                        </p>
                                        <p className="text-sm text-teal-700">
                                            {trimDescription(req.requirements)}
                                        </p>
                                    </div>
                                    <button
                                        tabIndex={-1}
                                        className="ml-auto flex-shrink-0 bg-teal-600 text-white px-3 py-1.5 rounded-lg hover:bg-teal-500 transition-colors text-xs flex items-center shadow-lg hover:shadow-teal-500/30"
                                    >
                                        <Eye className="w-3.5 h-3.5 mr-1 text-teal-300" />
                                        View
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
                <div className="flex-shrink-0 flex flex-col sm:flex-row justify-between items-center pt-4 border-t border-teal-200 gap-3">
                    <div className="text-sm text-teal-600">
                        Showing {Math.min(totalFilteredRequirements, startIdx + 1)} to{" "}
                        {Math.min(
                            totalFilteredRequirements,
                            startIdx + requirementsPerPage
                        )}{" "}
                        of {totalFilteredRequirements} files
                    </div>
                    <nav className="flex space-x-1">
                        <button
                            onClick={() => onPageChange(currentPage - 1)}
                            disabled={currentPage === 1}
                            className="px-3 py-1 rounded-lg border border-teal-300 text-sm font-medium text-teal-700 hover:bg-teal-100 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                        >
                            Previous
                        </button>
                        {pageNumbers.map((page, i) =>
                            page === "..." ? (
                                <span
                                    key={`ellipsis-${i}`}
                                    className="px-2 py-1 text-teal-400 self-end"
                                >
                                    …
                                </span>
                            ) : (
                                <button
                                    key={page}
                                    onClick={() => onPageChange(page)}
                                    className={`px-3 py-1 rounded-lg text-sm font-medium border shadow-lg ${
                                        currentPage === page
                                            ? "bg-teal-600 text-white border-teal-600"
                                            : "border-teal-300 text-teal-700 hover:bg-teal-100"
                                    }`}
                                >
                                    {page}
                                </button>
                            )
                        )}
                        <button
                            onClick={() => onPageChange(currentPage + 1)}
                            disabled={currentPage === totalPages}
                            className="px-3 py-1 rounded-lg border border-teal-300 text-sm font-medium text-teal-700 hover:bg-teal-100 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                        >
                            Next
                        </button>
                    </nav>
                </div>
            )}
        </div>
    );
}