import React from 'react';
import { ArrowLeft, BookOpen } from 'lucide-react';
import FileHierarchy from './FileHierarchy';

const ProjectSidebar = ({ 
  result, 
  selectedRequirement, 
  showProjectSummary, 
  onGoBack, 
  onShowProjectSummary, 
  onViewFile 
}) => {
  return (
    <div className="w-[30%] bg-white shadow-lg border-r border-gray-200 flex flex-col h-screen">
      {/* Header - Fixed */}
      <div className="p-6 border-b border-gray-200 flex-shrink-0">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-teal-800">
            Project Structure
          </h2>
          <button
            onClick={onGoBack}
            className="flex items-center px-3 py-1 bg-teal-100 hover:bg-teal-200 text-teal-800 rounded-lg transition-all duration-200 text-sm font-semibold"
            aria-label="Back to dashboard"
          >
            <ArrowLeft className="w-4 h-4 mr-1" />
            Dashboard
          </button>
        </div>
      </div>
      
      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          {/* Project Summary Button */}
          <div className="mb-4">
            <button
              onClick={onShowProjectSummary}
              className={`w-full flex items-center px-4 py-3 rounded-lg transition-all duration-200 font-medium text-left ${
                showProjectSummary 
                  ? 'bg-teal-100 text-teal-800 border-2 border-teal-200' 
                  : 'bg-gray-50 hover:bg-teal-50 text-gray-700 hover:text-teal-700 border-2 border-transparent hover:border-teal-100'
              }`}
            >
              <BookOpen className="w-5 h-5 mr-3 flex-shrink-0" />
              <span>Project Summary</span>
            </button>
          </div>

          {/* File Hierarchy */}
          <div className="border-t border-gray-200 pt-4">
            <h3 className="text-sm font-semibold text-gray-600 mb-3 uppercase tracking-wide">
              Files
            </h3>
            {result?.file_hierarchy ? (
              <FileHierarchy 
                fileHierarchy={result.file_hierarchy}
                onViewFile={onViewFile}
                selectedFile={selectedRequirement}
              />
            ) : (
              <div className="text-gray-500 text-center py-8 text-sm">
                No project structure available.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectSidebar;