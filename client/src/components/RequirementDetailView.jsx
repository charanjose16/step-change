import React from 'react';
import { ArrowLeft, Maximize2 } from 'lucide-react';
import MermaidGraph from './MermaidGraph';

const formatRequirementText = (text) => {
    return text
        .replace(/^(Overview|Objective|Use Case|Key Functionalities|Workflow Summary)$/gm, match => 
            `<strong class="text-teal-700 text-xl block mt-6 mb-3">${match}</strong>`
        )
        .replace(/\*\*/g, '')
        .replace(/\*/g, '')
        .replace(/^#+\s*/gm, '')
        .replace(/^-\s*/gm, '')
        .trim();
};

export default function RequirementDetailView({
  requirement,
  graphResponses,
  isLoadingGraph,
  graphError,
  selectedGraphIndex,
  onGraphIndexChange,
  onGoBack,
  onFullscreen
}) {
  const formattedRequirements = formatRequirementText(requirement?.requirements || '');

  return (
    <div className="flex flex-col w-full h-full p-6 overflow-y-auto bg-gray-50">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <button
            onClick={onGoBack}
            className="flex items-center px-4 py-2 bg-teal-100 hover:bg-teal-200 text-teal-800 rounded-lg transition-all duration-200 text-sm font-semibold shadow-lg hover:scale-105"
            aria-label="Go back to project structure"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back
          </button>
          <h2 className="text-2xl font-bold text-teal-800 ml-4">
            {requirement?.file_name || 'Unknown File'}
          </h2>
        </div>
      </div>

      <div className="mb-8 bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-teal-700 mb-3">File Path</h3>
        <p className="text-base text-teal-600">
          {requirement?.relative_path || 'No path available'}
        </p>
      </div>

      <div className="mb-8 bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-teal-700 mb-3">Full Requirements</h3>
        <div 
          className="prose max-w-full"
          dangerouslySetInnerHTML={{
            __html: formattedRequirements.split('\n\n').map(paragraph => 
              `<p class="mb-4 text-gray-800 leading-relaxed">${paragraph.trim()}</p>`
            ).join('')
          }}
        />
      </div>

      <div className="mb-8 bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-teal-700 mb-3">Diagrams</h3>
        {isLoadingGraph ? (
          <div className="flex items-center justify-center py-6 text-teal-600">
            <svg
              className="animate-spin h-8 w-8 mr-3"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
            Loading diagrams...
          </div>
        ) : graphError ? (
          <div className="text-red-600 text-base">{graphError}</div>
        ) : graphResponses.length === 0 ? (
          <div className="text-teal-600 text-base">No diagrams available</div>
        ) : (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <select
                value={selectedGraphIndex}
                onChange={(e) => onGraphIndexChange(Number(e.target.value))}
                className="px-4 py-2 border border-teal-300 rounded-lg text-base text-teal-800 focus:outline-none focus:ring-2 focus:ring-teal-500 bg-white shadow-sm"
              >
                {graphResponses.map((graph, index) => (
                  <option key={index} value={index}>
                    {graph.target_graph || `Diagram ${index + 1}`}
                  </option>
                ))}
              </select>
              <button
                onClick={onFullscreen}
                className="flex items-center px-4 py-2 bg-teal-600 hover:bg-teal-500 text-white rounded-lg text-base font-semibold shadow-lg hover:scale-105 transition-all duration-200"
                aria-label="View diagram in fullscreen"
              >
                <Maximize2 className="w-5 h-5 mr-2" />
                Fullscreen
              </button>
            </div>
            <div className="border border-teal-200 rounded-lg p-6 bg-white shadow-lg overflow-x-auto">
              <MermaidGraph chart={graphResponses[selectedGraphIndex]?.generated_code} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}