import React, { useEffect } from 'react';
import { ArrowLeft, Maximize2 } from 'lucide-react';
import MermaidGraph from './MermaidGraph';
import { useNavigate, useLocation } from 'react-router-dom';

const formatRequirementSummary = (text) => {
  if (!text) return '';

  // Split text into paragraphs based on double newlines
  const paragraphs = text.split('\n\n');
  let formatted = [];

  for (const paragraph of paragraphs) {
    // Split paragraph into lines and remove empty lines
    const lines = paragraph.split('\n').filter(l => l.trim());
    if (!lines.length) continue;

    // Skip any markdown/LLM-generated Dependent Files section
    const depFilesHeader = lines[0].match(/^#*\s*Dependent Files/i);
    // If this is a markdown/LLM Dependent Files list (with - ./ or - filename), skip this paragraph
    const isDepFilesList = depFilesHeader && lines.some(line => line.match(/^-\s*\.?\/?[\w\/-]+/));
    if (isDepFilesList) continue;

    // Check if paragraph is a numbered list (starts with "1.", "2.", etc.)
    const isNumberedList = lines[0].match(/^\d+\.\s+/);
    if (isNumberedList) {
      // Format each numbered list item properly inside <li>
      const listItems = lines
        .filter(line => line.match(/^\d+\.\s+/))
        .map(line => {
          const match = line.match(/^(\d+\.\s+)(.*)$/);
          const serial = match ? match[1] : '';
          const content = match ? match[2].trim() : line.trim();
          return `<li class="mb-1 text-sm text-gray-800">${serial} ${content}</li>`;
        });
      if (listItems.length) {
        formatted.push(`<ul class="list-decimal list-inside my-2">${listItems.join('')}</ul>`);
        continue;
      }
    }

    // Handle headers (Overview, Objective, Use Case, Key Functionalities, Workflow Summary), accepting Markdown-style and plain headers
    const headerMatch = lines[0].match(/^#*\s*(Overview|Objective|Use Case|Key Functionalities|Workflow Summary)\s*$/i);
    if (headerMatch) {
      const header = headerMatch[1].trim();
      const content = lines.slice(1).join(' ').trim();
      formatted.push(`<h3 class="text-teal-700 text-xl mt-6 mb-3 font-semibold">${header}</h3>`);
      if (content) {
        if (header.toLowerCase() === 'workflow summary' && /^1\./.test(content)) {
          // Split numbered steps into list items
          const steps = content.split(/\s*\d+\.\s+/).filter(Boolean);
          formatted.push('<ol class="list-decimal list-inside my-2">' + steps.map(step => `<li class="mb-1 text-sm text-gray-800">${step.trim()}</li>`).join('') + '</ol>');
        } else {
          formatted.push(`<p class="mb-3 text-sm text-gray-800 leading-relaxed">${content}</p>`);
        }
      }
    } else {
      // Regular paragraph with proper spacing and line-height
      const content = lines.join(' ').trim();
      formatted.push(`<p class="mb-3 text-sm text-gray-800 leading-relaxed">${content}</p>`);
    }
  }

  // Return the joined HTML string
  return formatted.join('');
};

export default function RequirementDetailView({
  requirement,
  graphResponses,
  isLoadingGraph,
  graphError,
  selectedGraphIndex,
  onGraphIndexChange,
  onGoBack,
  onFullscreen,
  allRequirements
}) {
  const navigate = useNavigate();
  const location = useLocation();
  const result = location.state?.result;

  useEffect(() => {
    console.log('Raw requirement:', requirement);
    console.log('Raw requirement.requirements:', requirement?.requirements);
    console.log('Dependencies:', requirement?.dependencies);
    console.log('All Requirements:', allRequirements);
    console.log('localStorage requirementsOutput:', localStorage.getItem('requirementsOutput'));
  }, [requirement, allRequirements]);

  const handleDependencyClick = (relativePath, fileName) => {
    console.log('Clicked:', { relativePath, fileName });
    let depRequirement = allRequirements.find(req => req.relative_path === relativePath);
    if (!depRequirement) {
      depRequirement = allRequirements.find(req => req.file_name === fileName);
    }
    console.log('Found depRequirement:', depRequirement);
    if (depRequirement) {
      navigate('/results', { 
        state: { 
          result: { 
            requirements: allRequirements, 
            file_hierarchy: result?.file_hierarchy || null 
          }, 
          selectedRequirement: depRequirement 
        } 
      });
    } else {
      console.error('No matching requirement found:', { relativePath, fileName });
    }
  };

  const parseDependentFiles = (text) => {
    const depSection = text.split('\n\n').find(p => p.trim().startsWith('Dependent Files'));
    if (!depSection) return [];

    const lines = depSection
      .replace(/\\n/g, '\n')
      .split('\n')
      .slice(1)
      .filter(line => line.trim());
    if (!lines.length || lines[0].trim() === 'No dependencies detected.') return [];

    let deps = [];
    // Handle numbered or bullet lists: e.g., '1. filename: description' or '- filename: description'
    for (const line of lines) {
      // Match numbered or bullet list
      const match = line.match(/^(?:\d+\.|-)\s*(.+?):\s*(.+)$/);
      if (match) {
        const file_name = match[1].trim();
        const overview = match[2].trim();
        deps.push({ file_name, overview });
      } else if (line.includes(':')) {
        // Fallback: 'filename: description'
        const [file_name, ...descParts] = line.split(':');
        deps.push({ file_name: file_name.trim(), overview: descParts.join(':').trim() });
      } else if (line.trim()) {
        // Fallback: just filename
        deps.push({ file_name: line.trim(), overview: '' });
      }
    }
    return deps;
  };

  const getDependencyOverview = (fileName) => {
    const depReq = allRequirements.find(req => req.file_name === fileName);
    if (!depReq?.requirements) return 'No overview available.';

    const sections = depReq.requirements.split('\n\n');
    const targetSection = sections.find(s => s.startsWith('Overview')) ||
                         sections.find(s => s.startsWith('Key Functionalities')) ||
                         sections[0] || '';
    
    const lines = targetSection
      .split('\n')
      .filter(l => l.trim() && !l.match(/^(Overview|Key Functionalities)$/))
      .slice(0, 3);
    
    const summary = lines.join(' ').substring(0, 200) + (lines.join(' ').length > 200 ? '...' : '');
    return summary || 'No specific overview available.';
  };

  const renderDependencies = () => {
    let dependencies = [];
    // Prefer explicit dependencies array from backend
    if (requirement?.dependencies && requirement.dependencies.length > 0) {
      dependencies = requirement.dependencies;
    } else {
      dependencies = parseDependentFiles(requirement?.requirements || '');
    }

    if (!dependencies.length) {
      return <p className="text-sm text-gray-600">No dependencies detected.</p>;
    }

    return (
      <div className="mt-4 space-y-3">
        {dependencies
          .map((dep, index) => {
            const fileName = dep.file_name || dep.filename || '';
            const matchedReq = allRequirements.find(req => req.file_name === fileName);
            const relPath = matchedReq?.relative_path || dep.relative_path || '';
            // Use getDependencyOverview to get a short overview for the file
            let overview = getDependencyOverview(fileName);
            // Fallback to dep.overview if getDependencyOverview returns empty
            if (!overview && dep.overview) overview = dep.overview;
            // Remove cards with technical import reason or 'No overview available.'
            if (
              !overview ||
              /^imports\b/i.test(overview) ||
              overview.toLowerCase().includes('providing components or utilities') ||
              overview.toLowerCase().includes('no overview available')
            ) {
              return null;
            }
            return (
              <div
                key={index}
                className={`mb-3 p-4 bg-white border border-gray-200 rounded-lg shadow-sm ${relPath ? 'cursor-pointer hover:bg-teal-50' : ''} transition-colors duration-200`}
                onClick={() => relPath && handleDependencyClick(relPath, fileName)}
              >
                <p className="font-semibold text-teal-600">{fileName}</p>
                {overview && (
                  <p className="text-sm text-gray-500 mt-1">{overview.replace(/^#+\s*/, '')}</p>
                )}
              </div>
            );
          })
          .filter(Boolean)}
      </div>
    );
  };

  const otherContent = formatRequirementSummary(
    requirement?.requirements
      ?.split('\n\n')
      .filter(p => !p.startsWith('Dependent Files'))
      .join('\n\n') || ''
  );

  return (
    <div className="flex flex-col w-full h-full p-6 overflow-y-auto bg-gray-50">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <button
            onClick={onGoBack}
            className="flex items-center px-4 py-2 bg-white text-teal-800 rounded-md transition-colors duration-200 text-sm font-semibold shadow-sm hover:bg-teal-100"
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

      <div className="mb-6 bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold text-teal-700 mb-2">File Path</h3>
        <p className="text-base text-gray-600">
          {requirement?.relative_path || 'No path available'}
        </p>
      </div>

      <div className="mb-6 bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold text-teal-700 mb-2">Full Requirements</h3>
        <div className="prose max-w-full text-gray-800">
          <div dangerouslySetInnerHTML={{ __html: otherContent }} />
          <div className="mt-8">
            <h3 className="text-xl font-semibold text-teal-700 mb-4">Dependent Files</h3>
            {renderDependencies()}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold text-teal-700 mb-2">Diagrams</h3>
        {isLoadingGraph ? (
          <div className="flex items-center justify-center py-6 text-gray-600">
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
            Loading Diagrams...
          </div>
        ) : graphError ? (
          <div className="text-red-600 text-base">{graphError}</div>
        ) : graphResponses.length === 0 ? (
          <div className="text-gray-600 text-base">No diagrams available</div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <select
                value={selectedGraphIndex}
                onChange={(e) => onGraphIndexChange(Number(e.target.value))}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-teal-500 bg-white shadow-sm"
              >
                {graphResponses.map((graph, index) => (
                  <option key={index} value={index}>
                    {graph?.target_graph || `Diagram ${index + 1}`}
                  </option>
                ))}
              </select>
              <button
                onClick={onFullscreen}
                className="flex items-center px-3 py-2 bg-teal-600 text-white rounded-md text-sm font-semibold shadow-sm hover:bg-teal-700 transition-colors"
                aria-label="View diagram in fullscreen"
              >
                <Maximize2 className="w-4 h-4 mr-1" />
                Fullscreen
              </button>
            </div>
            <div className="border border-gray-200 rounded-md p-4 bg-white shadow-sm overflow-x-auto">
              <MermaidGraph chart={graphResponses[selectedGraphIndex]?.generated_code || ''} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}