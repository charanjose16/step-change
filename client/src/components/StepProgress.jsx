import React from 'react';

const steps = [
  {
    id: 1,
    label: "Fetching from S3",
    description: "Downloading project files from the bucket.",
    iconPath: "/src/assets/icons/1.png",
  },
  {
    id: 2,
    label: "Analyzing Files",
    description: "Parsing and understanding code structure.",
    iconPath: "/src/assets/icons/2.png",
  },
  {
    id: 3,
    label: "Generating Business Logic",
    description: "Creating relevant use-case and logic blocks.",
    iconPath: "/src/assets/icons/3.png",
  },
  {
    id: 4,
    label: "Finalizing",
    description: "Wrapping up output and results.",
    iconPath: "/src/assets/icons/4.png",
  },
];

const StepProgress = ({ loadingProgress = 0 }) => {
  const currentStep = Math.min(Math.floor(loadingProgress / 25) + 1, steps.length);
  const isComplete = loadingProgress >= 100;

  return (
    <div className="mt-6 p-6 bg-teal-50 rounded-2xl border border-teal-200 shadow-sm">
      <h3 className="text-2xl font-bold text-teal-800 mb-6 tracking-wide">
        {isComplete ? "Generation Complete!" : "Generation in Progress"}
      </h3>

      <div className="relative pl-10">
        {steps.map((step, index) => {
          const isCompleted = step.id < currentStep || isComplete;
          const isActive = step.id === currentStep && !isComplete;

          return (
            <div key={step.id} className="relative">
              {/* Connector Line */}
              {index < steps.length - 1 && (
                <div
                  className={`absolute left-[22px] top-12 w-[4px] h-[64px] rounded-full z-0 ${
                    isCompleted ? "bg-green-500" : "bg-gray-300"
                  }`}
                />
              )}

              {/* Step */}
              <div
                className={`relative z-10 flex items-center gap-4 py-4 ${
                  isCompleted
                    ? "text-green-800"
                    : isActive
                    ? "text-teal-800"
                    : "text-gray-400"
                }`}
              >
                <div
                  className={`w-12 h-12 flex items-center justify-center rounded-full border-2 shadow-md ${
                    isCompleted
                      ? "bg-green-100 border-green-600"
                      : isActive
                      ? "bg-teal-100 border-teal-600 animate-pulse"
                      : "bg-gray-100 border-gray-300"
                  }`}
                >
                  <img
                    src={step.iconPath}
                    alt={step.label}
                    className="w-6 h-6 object-contain"
                  />
                </div>

                <div>
                  <div className="text-base font-semibold">{step.label}</div>
                  <div className="text-sm text-gray-500">
                    {step.description}
                  </div>
                </div>
              </div>
            </div>
          );
        })}

        {/* Progress Text */}
        <div className="mt-4 text-right text-sm text-teal-700 font-semibold pr-2">
          Progress: {loadingProgress.toFixed(2)}%
        </div>
      </div>
    </div>
  );
};

export default StepProgress;
