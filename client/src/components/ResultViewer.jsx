import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "../api/axiosConfig";
import FullScreenGraph from "./FullScreenGraph";
import RequirementDetailView from "./RequirementDetailView";
import { Loader2, AlertTriangle, ArrowLeft, FileText, Target, Building, Layers, Zap, Workflow, TrendingUp, Code, ChevronDown, BarChart3, Maximize2, BookOpen, Minimize2 } from "lucide-react";
import FileHierarchy from './FileHierarchy';
import MermaidGraph from './MermaidGraph';

const GraphDropdown = ({ summaryGraphResponses, loadingSummaryGraphs, summaryGraphError, onFullscreen }) => {
  const [selectedGraphIndex, setSelectedGraphIndex] = useState(0);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  

  if (loadingSummaryGraphs) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
        <div className="flex items-center mb-4">
          <BarChart3 className="h-6 w-6 text-teal-600 mr-2" />
          <h2 className="text-2xl font-bold text-teal-800">Project Graphs</h2>
        </div>
        <div className="flex items-center justify-center py-8 text-teal-600">
          <Loader2 className="h-5 w-5 animate-spin mr-2" />
          Loading graphs...
        </div>
      </div>
    );
  }

  if (summaryGraphError) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
        <div className="flex items-center mb-4">
          <BarChart3 className="h-6 w-6 text-teal-600 mr-2" />
          <h2 className="text-2xl font-bold text-teal-800">Project Graphs</h2>
        </div>
        <div className="flex items-center justify-center py-8 text-red-600">
          <AlertTriangle className="h-5 w-5 mr-2" />
          {summaryGraphError}
        </div>
      </div>
    );
  }

  if (!summaryGraphResponses || summaryGraphResponses.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
        <div className="flex items-center mb-4">
          <BarChart3 className="h-6 w-6 text-teal-600 mr-2" />
          <h2 className="text-2xl font-bold text-teal-800">Project Graphs</h2>
        </div>
        <div className="text-gray-500 text-center py-8 italic">
          No graphs available for this project.
        </div>
      </div>
    );
  }

  const selectedGraph = summaryGraphResponses[selectedGraphIndex];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <BarChart3 className="h-6 w-6 text-teal-600 mr-2" />
          <h2 className="text-2xl font-bold text-teal-800">Project Graphs</h2>
        </div>
        
        {/* Graph Selection Dropdown */}
        <div className="relative">
          <button
            onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            className="flex items-center px-4 py-2 bg-teal-50 hover:bg-teal-100 text-teal-800 rounded-lg transition-colors duration-200 border border-teal-200"
          >
            <span className="mr-2 font-medium">
              {selectedGraph?.target_graph || `Graph ${selectedGraphIndex + 1}`}
            </span>
            <ChevronDown className={`h-4 w-4 transition-transform duration-200 ${isDropdownOpen ? 'rotate-180' : ''}`} />
          </button>

          {isDropdownOpen && (
            <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-xl border border-gray-200 z-10">
              <div className="py-1">
                {summaryGraphResponses.map((graph, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setSelectedGraphIndex(index);
                      setIsDropdownOpen(false);
                    }}
                    className={`w-full text-left px-4 py-2 hover:bg-teal-50 transition-colors duration-150 ${
                      index === selectedGraphIndex ? 'bg-teal-50 text-teal-800' : 'text-gray-700'
                    }`}
                  >
                    <div className="font-medium">
                      {graph.target_graph || `Graph ${index + 1}`}
                    </div>
                    {graph.description && (
                      <div className="text-sm text-gray-500 mt-1">
                        {graph.description}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Graph Display */}
      <div className="relative">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">
            {selectedGraph?.target_graph || `Graph ${selectedGraphIndex + 1}`}
          </h3>
          <button
            onClick={() => {
              if (onFullscreen && selectedGraph?.generated_code) {
                onFullscreen(selectedGraph.generated_code);
              }
            }}
            className="flex items-center px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors duration-200 text-sm"
            disabled={!selectedGraph?.generated_code}
          >
            <Maximize2 className="h-4 w-4 mr-1" />
            Fullscreen
          </button>
        </div>
        
        {selectedGraph?.generated_code ? (
          <div className="border border-gray-200 rounded-lg overflow-hidden">
            <MermaidGraph chart={String(selectedGraph.generated_code)} />
          </div>
        ) : (
          <div className="text-gray-500 text-center py-8 italic border border-gray-200 rounded-lg">
            No graph data available
          </div>
        )}
        
        {selectedGraph?.description && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-800 mb-2">Description:</h4>
            <p className="text-gray-600 text-sm">{selectedGraph.description}</p>
          </div>
        )}
      </div>
    </div>
  );
};

const ProjectSummaryDisplay = ({ projectSummary, loadingSummary, summaryGraphResponses, loadingSummaryGraphs, summaryGraphError, onFullscreen }) => {
  const parseProjectSummary = (summary) => {
    if (!summary || typeof summary !== 'string') return null;

    const sections = [];
    const lines = summary.split('\n').filter(line => line.trim());
    
    let currentSection = null;
    
    for (const line of lines) {
      const trimmedLine = line.trim();
      
      if (trimmedLine.startsWith('## ')) {
        if (currentSection) {
          sections.push(currentSection);
        }
        const title = trimmedLine.replace('## ', '');
        currentSection = {
          title,
          content: '',
          key: title.toLowerCase().replace(/[^a-z0-9]/g, '_')
        };
      } else if (currentSection && trimmedLine) {
        currentSection.content += (currentSection.content ? ' ' : '') + trimmedLine;
      }
    }
    
    if (currentSection) {
      sections.push(currentSection);
    }
    
    return sections.length > 0 ? sections : null;
  };

  const getSectionIcon = (title) => {
    const titleLower = title.toLowerCase();
    if (titleLower.includes('overview')) return <FileText className="h-5 w-5" />;
    if (titleLower.includes('business') && titleLower.includes('context')) return <Target className="h-5 w-5" />;
    if (titleLower.includes('architecture')) return <Layers className="h-5 w-5" />;
    if (titleLower.includes('capabilities') || titleLower.includes('features')) return <Zap className="h-5 w-5" />;
    if (titleLower.includes('workflow') || titleLower.includes('process')) return <Workflow className="h-5 w-5" />;
    if (titleLower.includes('value') || titleLower.includes('impact')) return <TrendingUp className="h-5 w-5" />;
    if (titleLower.includes('technical')) return <Code className="h-5 w-5" />;
    return <Building className="h-5 w-5" />;
  };

  if (loadingSummary) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center mb-4">
          <FileText className="h-6 w-6 text-teal-600 mr-2" />
          <h1 className="text-2xl font-bold text-teal-800">Project Summary</h1>
        </div>
        <div className="flex items-center justify-center py-8 text-teal-600">
          <Loader2 className="h-5 w-5 animate-spin mr-2" />
          Generating project summary...
        </div>
      </div>
    );
  }

  if (!projectSummary) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center mb-4">
          <FileText className="h-6 w-6 text-teal-600 mr-2" />
          <h1 className="text-2xl font-bold text-teal-800">Project Summary</h1>
        </div>
        <div className="text-teal-600 text-center py-4 italic">
          No project summary available. The analysis may still be processing or no summary was generated.
        </div>
      </div>
    );
  }

  const sections = parseProjectSummary(projectSummary);

  return (
    <div className="space-y-6">
      {/* Project Summary Section */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center mb-6">
          <FileText className="h-6 w-6 text-teal-600 mr-2" />
          <h1 className="text-2xl font-bold text-teal-800">Project Summary</h1>
        </div>
        
        {sections ? (
          <div className="space-y-4">
            {sections.map((section, ) => (
              <div key={section.key} className="border-l-4 border-teal-200 pl-4">
                <div className="flex items-center space-x-3 mb-2">
                  <div className="text-teal-600">
                    {getSectionIcon(section.title)}
                  </div>
                  <h3 className="text-lg font-semibold text-gray-800">
                    {section.title}
                  </h3>
                </div>
                <div className="text-gray-700 leading-relaxed">
                  {section.content}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-gray-700 leading-relaxed">
            {projectSummary}
          </div>
        )}
      </div>

      {/* Project Graphs Section */}
      <GraphDropdown
        summaryGraphResponses={summaryGraphResponses}
        loadingSummaryGraphs={loadingSummaryGraphs}
        summaryGraphError={summaryGraphError}
        onFullscreen={onFullscreen}
      />
    </div>
  );
};

import ProjectVisualizationModal from './ProjectVisualizationModal';


export default function ResultViewer() {
    const [loadedRequirements, setLoadedRequirements] = useState(null);
    const [showProjectGraphModal, setShowProjectGraphModal] = useState(false);
    const [loadError, setLoadError] = useState("");
    const [selectedRequirement, setSelectedRequirement] = useState(null);
    const [showProjectSummary, setShowProjectSummary] = useState(false);
    const [graphResponses, setGraphResponses] = useState([]);
    const [loadingGraph, setLoadingGraph] = useState(false);
    const [graphError, setGraphError] = useState("");
    const [isFullScreen, setIsFullScreen] = useState(false);
    const [fullScreenChart, setFullScreenChart] = useState(null);
    const [detailGraphIndex, setDetailGraphIndex] = useState(0);
    const [projectSummary, setProjectSummary] = useState(null);
    const [loadingSummary, setLoadingSummary] = useState(false);
    const [summaryGraphResponses, setSummaryGraphResponses] = useState([]);
    const [loadingSummaryGraphs, setLoadingSummaryGraphs] = useState(false);
    const [summaryGraphError, setSummaryGraphError] = useState("");
    const [isRightSectionFullscreen, setIsRightSectionFullscreen] = useState(false);

    const location = useLocation();
    const navigate = useNavigate();
    const result = location.state?.result;
    const initialSelectedRequirement = location.state?.selectedRequirement;



    useEffect(() => {
        if (!result) {
            setLoadError("No analysis results available. Please analyze a codebase first.");
            return;
        }

        setLoadError("");
        setLoadedRequirements(null);
        setSelectedRequirement(initialSelectedRequirement || null);
        setShowProjectSummary(false);
        setGraphResponses([]);
        setGraphError("");
        setIsFullScreen(false);
        setFullScreenChart(null);
        setDetailGraphIndex(0);
        setProjectSummary(null);
        setSummaryGraphResponses([]);
        setSummaryGraphError("");

        if (result && (Array.isArray(result.requirements) || Array.isArray(result.files))) {
            console.log("Using result prop:", result);
            setLoadedRequirements({ 
                files: result.requirements || result.files || []
            });
            if (result.project_summary) {
                setProjectSummary(result.project_summary);
            }
            return;
        }

        const storedResult = localStorage.getItem("requirementsOutput");
        if (storedResult) {
            try {
                const parsedResult = JSON.parse(storedResult);
                console.log("Parsed stored result:", parsedResult);
                
                if (parsedResult) {
                    let files = [];
                    if (Array.isArray(parsedResult.requirements)) {
                        files = parsedResult.requirements;
                    } else if (Array.isArray(parsedResult.files)) {
                        files = parsedResult.files;
                    } else if (Array.isArray(parsedResult)) {
                        files = parsedResult;
                    }
                    
                    setLoadedRequirements({ files: files });
                    if (parsedResult.project_summary) {
                        setProjectSummary(parsedResult.project_summary);
                    }
                } else {
                    console.error("Stored data format invalid:", parsedResult);
                    setLoadError("Failed to load results: Invalid data format.");
                    localStorage.removeItem("requirementsOutput");
                }
            } catch (e) {
                console.error("Failed to parse stored results:", e);
                setLoadError("Failed to load results: Could not parse data.");
                localStorage.removeItem("requirementsOutput");
            }
        } else {
            setLoadError("No analysis results found in storage.");
        }
    }, [result, initialSelectedRequirement]);

    useEffect(() => {
        const generateFallbackSummary = () => {
            if (!loadedRequirements || projectSummary || loadingSummary) {
                return;
            }

            setLoadingSummary(true);
            
            try {
                const filesWithRequirements = loadedRequirements.files.filter(
                    file => file.requirements && file.requirements.trim()
                );
                
                if (filesWithRequirements.length === 0) {
                    setProjectSummary("No valid requirements found to generate a project summary.");
                    setLoadingSummary(false);
                    return;
                }

                const req = localStorage.getItem('requirementsOutput');
                const reqJson = JSON.parse(req);
                const summary = reqJson["summary"];
                
                setProjectSummary(summary);
            } catch (error) {
                console.error("Error generating fallback summary:", error);
                setProjectSummary("Unable to generate project summary due to an error.");
            } finally {
                setLoadingSummary(false);
            }
        };

        const timer = setTimeout(generateFallbackSummary, 1000);
        return () => clearTimeout(timer);
    }, [loadedRequirements, projectSummary, loadingSummary]);

    useEffect(() => {
        const fetchGraphs = async () => {
            if (!selectedRequirement?.requirements?.trim()) {
                setGraphResponses([]);
                setGraphError(
                    selectedRequirement ? "No requirements text found for this file." : ""
                );
                setLoadingGraph(false);
                return;
            }

            setLoadingGraph(true);
            setGraphResponses([]);
            setGraphError("");
            try {
                const payload = { requirement: selectedRequirement.requirements };
                const response = await axios.post("/analysis/graphs", payload);
                const data = Array.isArray(response.data) ? response.data : [];
                const validGraphs = data.filter((g) => g && g.generated_code) || [];

                setGraphResponses(validGraphs);
                setDetailGraphIndex(0);

                if (validGraphs.length === 0) {
                    setGraphError("No valid graphs were generated by the analysis.");
                }
            } catch (error) {
                console.error("Error fetching graphs:", error);
                const errorMsg =
                    error.response?.data?.detail || error.message || "Server error";
                setGraphError(`Failed to fetch graphs: ${errorMsg}`);
            } finally {
                setLoadingGraph(false);
            }
        };

        if (selectedRequirement) {
            fetchGraphs();
        } else {
            setGraphResponses([]);
            setGraphError("");
            setLoadingGraph(false);
            setDetailGraphIndex(0);
        }
    }, [selectedRequirement]);

    useEffect(() => {
        const fetchSummaryGraphs = async () => {
            if (!projectSummary?.trim() || !showProjectSummary) {
                setSummaryGraphResponses([]);
                setSummaryGraphError("");
                setLoadingSummaryGraphs(false);
                return;
            }

            setLoadingSummaryGraphs(true);
            setSummaryGraphResponses([]);
            setSummaryGraphError("");
            try {
                const payload = { requirement: projectSummary };
                const response = await axios.post("/analysis/graphs", payload);
                const data = Array.isArray(response.data) ? response.data : [];
                const validGraphs = data.filter((g) => g && g.generated_code) || [];

                setSummaryGraphResponses(validGraphs);

                if (validGraphs.length === 0) {
                    setSummaryGraphError("No valid graphs were generated for the project summary.");
                }
            } catch (error) {
                console.error("Error fetching summary graphs:", error);
                const errorMsg =
                    error.response?.data?.detail || error.message || "Server error";
                setSummaryGraphError(`Failed to fetch summary graphs: ${errorMsg}`);
            } finally {
                setLoadingSummaryGraphs(false);
            }
        };

        if (showProjectSummary && projectSummary) {
            fetchSummaryGraphs();
        }
    }, [projectSummary, showProjectSummary]);

    const handleGoBack = () => {
        navigate('/dashboard');
    };

    const handleGraphIndexChange = (newIndex) => {
        setDetailGraphIndex(newIndex);
    };

    const handleFullscreen = (chartData = null) => {
        if (chartData && typeof chartData === 'string') {
            setFullScreenChart(chartData);
            setIsFullScreen(true);
        } else if (graphResponses?.[detailGraphIndex]?.generated_code) {
            setFullScreenChart(String(graphResponses[detailGraphIndex].generated_code));
            setIsFullScreen(true);
        }
    };

    const handleRightSectionFullscreen = () => {
        setIsRightSectionFullscreen(!isRightSectionFullscreen);
    };

    const handleViewFile = (file) => {
        const requirement = result?.requirements?.find(
            req => req.file_name === file.name || req.relative_path === file.path
        );
        if (requirement) {
            setSelectedRequirement(requirement);
            setShowProjectSummary(false);
        }
    };

    const handleShowProjectSummary = () => {
        setSelectedRequirement(null);
        setShowProjectSummary(true);
    };
    
function handleShowProjectVisualization() {
  setShowProjectGraphModal(true);
}

    const handleClearSelection = () => {
        setSelectedRequirement(null);
        setShowProjectSummary(false);
    };

    let content = null;
    if (loadError) {
        content = (
            <div className="flex items-center justify-center h-screen text-red-600">
                <AlertTriangle className="h-8 w-8 mr-3" />
                <span>{loadError}</span>
            </div>
        );
    } else if (!loadedRequirements) {
        content = (
            <div className="flex items-center justify-center h-screen text-teal-600">
                <Loader2 className="h-8 w-8 animate-spin mr-3" />
                <span>Loading results...</span>
            </div>
        );
    } else {
        content = (
            <div className="h-screen flex bg-gradient-to-br from-teal-50 to-teal-100 overflow-hidden transition-all duration-200">
                {/* Left Sidebar - File Hierarchy (30%) - Hide in fullscreen mode */}
                {!isRightSectionFullscreen && (
                    <div className="w-[30%] bg-white shadow-lg border-r border-gray-200 flex flex-col h-full">
                        {/* Fixed Header */}
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
                        
                        {/* Scrollable Content */}
                        <div className="flex-1 overflow-y-auto p-4">
                            {/* Project Summary Button */}
                            <div className="mb-4 flex flex-col gap-3">
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
                                <button
                                    onClick={handleShowProjectVisualization}
                                    className="w-full flex items-center px-4 py-3 rounded-lg transition-all duration-200 font-medium text-left bg-teal-50 hover:bg-teal-100 text-teal-700 border-2 border-transparent hover:border-teal-200"
                                >
                                    <BarChart3 className="w-5 h-5 mr-3 flex-shrink-0" />
                                    <span>Project Visualization</span>
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
                )}

                {/* Right Content Area (70% or 100% in fullscreen) */}
                <div className={`${isRightSectionFullscreen ? 'w-full' : 'w-[70%]'} flex flex-col h-full overflow-hidden transition-all duration-300`}>
                    {/* Fixed Header */}
                    <div className="bg-white shadow-sm border-b border-gray-200 p-6 flex-shrink-0" style={{ minHeight: '88px' }}>
                        <div className="flex items-center justify-between">
                            <h1 className="text-2xl font-bold text-gray-800 min-w-0 flex-1">
                                {showProjectSummary ? 'Project Summary' : 
                                 selectedRequirement ? selectedRequirement.file_name || selectedRequirement.relative_path : 
                                 'Welcome to Project Analysis'}
                            </h1>
                            <div className="flex items-center gap-2 ml-4">
                                {/* Fullscreen Toggle Button */}
                                <button
                                    onClick={handleRightSectionFullscreen}
                                    className="flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-all duration-200 text-sm font-semibold"
                                    title={isRightSectionFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
                                >
                                    {isRightSectionFullscreen ? (
                                        <>
                                            <Minimize2 className="w-4 h-4 mr-1" />
                                            Exit Fullscreen
                                        </>
                                    ) : (
                                        <>
                                            <Maximize2 className="w-4 h-4 mr-1" />
                                            Fullscreen
                                        </>
                                    )}
                                </button>
                                
                                {/* Back Button */}
                                {(selectedRequirement || showProjectSummary) && (
                                    <button
                                        onClick={handleClearSelection}
                                        className="flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-all duration-200 text-sm font-semibold"
                                    >
                                        <ArrowLeft className="w-4 h-4 mr-1" />
                                        Back
                                    </button>
                                )}
                                
                                {/* Dashboard Button - Only show in fullscreen mode */}
                                {isRightSectionFullscreen && (
                                    <button
                                        onClick={handleGoBack}
                                        className="flex items-center px-3 py-2 bg-teal-100 hover:bg-teal-200 text-teal-800 rounded-lg transition-all duration-200 text-sm font-semibold"
                                        aria-label="Back to dashboard"
                                    >
                                        <ArrowLeft className="w-4 h-4 mr-1" />
                                        Dashboard
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Scrollable Content with fixed height to prevent layout shifts */}
                    <div className="flex-1 overflow-y-auto bg-gray-50" style={{ minHeight: 'calc(100vh - 120px)' }}>
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
                                            {isRightSectionFullscreen 
                                                ? "Exit fullscreen to access the project structure or use the buttons above to navigate."
                                                : "Select a file from the project structure or view the project summary to get started with your analysis."
                                            }
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
        );
    }

    return (
        <div className="h-screen overflow-hidden bg-gray-50">
            {content}
            {/* Project Visualization Modal */}
            <ProjectVisualizationModal
                show={showProjectGraphModal}
                onClose={() => setShowProjectGraphModal(false)}
                folderName={result?.folder_name || (JSON.parse(localStorage.getItem('requirementsOutput'))?.folder_name)}
            />
            {isFullScreen && fullScreenChart && (
                <FullScreenGraph
                    chart={fullScreenChart}
                    onClose={() => {
                        setIsFullScreen(false);
                        setFullScreenChart(null);
                    }}
                    className="bg-teal-700 text-white shadow-lg"
                />
            )}
        </div>
    );
}
