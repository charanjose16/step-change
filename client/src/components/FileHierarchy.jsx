import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, File, Folder } from 'lucide-react';

const FileHierarchyItem = ({ 
  item, 
  depth = 0, 
  onViewFile,
  selectedFile,
  isRoot = false
}) => {
  const [isExpanded, setIsExpanded] = useState(isRoot);

  useEffect(() => {
    if (isRoot) {
      setIsExpanded(true);
    }
  }, [isRoot]);

  const toggleExpand = () => {
    if (item.type === 'directory') {
      setIsExpanded(!isExpanded);
    } else if (item.type === 'file') {
      onViewFile(item);
    }
  };

  const isSelected = selectedFile && (
    selectedFile.file_name === item.name || 
    selectedFile.relative_path === item.path
  );

  const renderIcon = () => {
    if (item.type === 'directory') {
      return isExpanded ? (
        <ChevronDown className="w-4 h-4 text-teal-600 flex-shrink-0" />
      ) : (
        <ChevronRight className="w-4 h-4 text-teal-600 flex-shrink-0" />
      );
    }
    return <File className="w-4 h-4 text-teal-500 flex-shrink-0" />;
  };

  return (
    <div>
      <div 
        className={`flex items-center cursor-pointer py-2 px-3 rounded-lg transition-all duration-200 hover:bg-teal-50 ${
          isSelected ? 'bg-teal-100 border-2 border-teal-200' : ''
        } ${
          item.type === 'directory' ? 'font-medium text-gray-700' : 'text-gray-600'
        }`}
        style={{ marginLeft: `${depth * 16}px` }}
        onClick={toggleExpand}
      >
        <span className="mr-2">{renderIcon()}</span>
        <span className="flex-grow text-sm truncate" title={item.name}>
          {item.name}
        </span>
        {item.type === 'directory' && (
          <Folder className="w-4 h-4 text-teal-400 ml-2 flex-shrink-0" />
        )}
      </div>
      
      {item.type === 'directory' && isExpanded && item.children && (
        <div className="mt-1">
          {item.children.length > 0 ? (
            item.children.map((child, index) => (
              <FileHierarchyItem 
                key={`${child.name}-${index}`} 
                item={child} 
                depth={depth + 1}
                onViewFile={onViewFile}
                selectedFile={selectedFile}
                isRoot={false}
              />
            ))
          ) : (
            <div className="text-gray-400 text-xs pl-6 py-1" style={{ marginLeft: `${(depth + 1) * 16}px` }}>
              Empty folder
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Updated ResultViewer component sections for proper scrolling
const ResultViewer = () => {
  // ... existing state and effects ...

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Main content */}
      <div className="flex h-screen bg-gradient-to-br from-teal-50 to-teal-100">
        {/* Left Sidebar - File Hierarchy (30%) */}
        <div className="w-[30%] bg-white shadow-lg border-r border-gray-200 flex flex-col h-full">
          <div className="p-6 border-b border-gray-200 flex-shrink-0">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-teal-800">
                Project Structure
              </h2>
              <button
                onClick={handleGoBack}
                className="flex items-center px-3 py-1 bg-teal-100 hover:bg-teal-200 text-teal-800 rounded-lg transition-all duration-200 text-sm font-semibold"
                aria-label="Back to dashboard"
              >
                <ArrowLeft className="w-4 h-4 mr-1" />
                Dashboard
              </button>
            </div>
          </div>
          
          <div className="flex-1 overflow-hidden flex flex-col">
            {/* Project Summary Button */}
            <div className="p-4 flex-shrink-0">
              <button
                onClick={handleShowProjectSummary}
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

            {/* File Hierarchy - Scrollable */}
            <div className="flex-1 overflow-hidden flex flex-col">
              <div className="px-4 pb-2 flex-shrink-0">
                <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide border-t border-gray-200 pt-4">
                  Files
                </h3>
              </div>
              <div className="flex-1 overflow-y-auto px-4 pb-4">
                {result?.file_hierarchy ? (
                  <FileHierarchy 
                    fileHierarchy={result.file_hierarchy}
                    onViewFile={handleViewFile}
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

        {/* Right Content Area (70%) */}
        <div className="flex-1 flex flex-col overflow-hidden h-full">
          <div className="bg-white shadow-sm border-b border-gray-200 p-6 flex-shrink-0">
            <div className="flex items-center justify-between">
              <h1 className="text-2xl font-bold text-gray-800">
                {showProjectSummary ? 'Project Summary' : 
                 selectedRequirement ? selectedRequirement.file_name || selectedRequirement.relative_path : 
                 'Welcome to Project Analysis'}
              </h1>
              {(selectedRequirement || showProjectSummary) && (
                <button
                  onClick={handleClearSelection}
                  className="flex items-center px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-all duration-200 text-sm font-semibold"
                >
                  <ArrowLeft className="w-4 h-4 mr-1" />
                  Back
                </button>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto bg-gray-50">
            {/* Content area with proper scrolling */}
            {showProjectSummary ? (
              <div className="p-8">
                <ProjectSummaryDisplay
                  projectSummary={projectSummary}
                  loadingSummary={loadingSummary}
                  summaryGraphResponses={summaryGraphResponses}
                  loadingSummaryGraphs={loadingSummaryGraphs}
                  summaryGraphError={summaryGraphError}
                  onFullscreen={handleFullscreen}
                />
              </div>
            ) : selectedRequirement ? (
              <div className="h-full">
                <RequirementDetailView
                  requirement={selectedRequirement}
                  graphResponses={graphResponses}
                  isLoadingGraph={loadingGraph}
                  graphError={graphError}
                  selectedGraphIndex={detailGraphIndex}
                  onGraphIndexChange={handleGraphIndexChange}
                  onGoBack={handleClearSelection}
                  onFullscreen={handleFullscreen}
                  allRequirements={result?.requirements || []}
                />
              </div>
            ) : (
              <div className="p-8">
                <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                  <div className="mb-6">
                    <FileText className="h-16 w-16 text-teal-600 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">
                      Ready to Explore
                    </h2>
                    <p className="text-gray-600 max-w-md mx-auto">
                      Select a file from the project structure or view the project summary to get started with your analysis.
                    </p>
                  </div>
                  
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <button
                      onClick={handleShowProjectSummary}
                      className="flex items-center px-6 py-3 bg-teal-600 hover:bg-teal-700 text-white rounded-lg transition-colors duration-200 font-medium"
                    >
                      <BookOpen className="w-5 h-5 mr-2" />
                      View Project Summary
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default function FileHierarchy({ 
  fileHierarchy, 
  onViewFile,
  selectedFile
}) {
  return (
    <div className="w-full h-full overflow-y-auto overflow-x-hidden">
      <div className="pr-2"> {/* Add padding for scrollbar */}
        {fileHierarchy ? (
          <FileHierarchyItem 
            item={fileHierarchy} 
            onViewFile={onViewFile}
            selectedFile={selectedFile}
            isRoot={true}
          />
        ) : (
          <div className="text-gray-500 text-center py-8 text-sm">
            No project structure available
          </div>
        )}
      </div>
    </div>
  );
}