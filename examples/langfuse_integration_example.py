"""
Example: Integrating Langfuse Observability into Mobile Crash RCA Agent

This example demonstrates how to add comprehensive observability to your
mobile crash analysis agent using Langfuse.
"""

import os
import sys
import time
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from observability import LangfuseHandler, ObservabilityConfig
from observability.decorators import set_global_handler, trace_llm_call, trace_agent_step
from llm.gpt_client import GPTClient


class ObservableMobileCrashAgent:
    """Mobile Crash RCA Agent with Langfuse observability"""
    
    def __init__(self):
        """Initialize the agent with observability"""
        # Setup observability - load from config.yml first, then environment
        self.obs_config = ObservabilityConfig.from_config_file()
        self.obs_handler = LangfuseHandler(self.obs_config)
        
        # Set global handler for decorators
        set_global_handler(self.obs_handler)
        
        # Initialize LLM client with environment variables
        self.llm_client = GPTClient(
            model_name=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            temperature=0.1
        )
        
        print(f"✅ Agent initialized with observability: {self.obs_config.langfuse_enabled}")
    
    def analyze_crash_data(self, crash_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze crash data with full observability"""
        
        # Create a session trace for the entire analysis
        with self.obs_handler.trace_session(
            name="mobile_crash_analysis",
            user_id=crash_data.get("user_id", "system"),
            app_version=crash_data.get("app_version"),
            device_info=crash_data.get("device_info"),
            crash_type=crash_data.get("crash_type")
        ) as trace:
            
            try:
                # Step 1: Data preprocessing
                processed_data = self._preprocess_crash_data(crash_data)
                
                # Step 2: Pattern analysis
                patterns = self._analyze_patterns(processed_data)
                
                # Step 3: Root cause analysis
                root_causes = self._identify_root_causes(patterns)
                
                # Step 4: Generate recommendations
                recommendations = self._generate_recommendations(root_causes)
                
                # Step 5: Quality evaluation
                quality_score = self._evaluate_analysis_quality(recommendations)
                
                # Add quality score to Langfuse
                self.obs_handler.add_score("analysis_quality", quality_score, "AI-evaluated quality score")
                
                result = {
                    "success": True,
                    "patterns": patterns,
                    "root_causes": root_causes,
                    "recommendations": recommendations,
                    "quality_score": quality_score,
                    "metadata": {
                        "processing_time": time.time(),
                        "crash_id": crash_data.get("crash_id"),
                        "session_id": trace.id if trace else None
                    }
                }
                
                return result
                
            except Exception as e:
                self.obs_handler.log_error(e, "crash_analysis_session")
                return {
                    "success": False,
                    "error": str(e),
                    "session_id": trace.id if trace else None
                }
    
    @trace_agent_step(name="preprocess_crash_data")
    def _preprocess_crash_data(self, crash_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess crash data"""
        # Simulate preprocessing
        time.sleep(0.1)
        
        processed = {
            "clean_stack_trace": crash_data.get("stack_trace", "").strip(),
            "device_category": self._categorize_device(crash_data.get("device_info", {})),
            "crash_frequency": crash_data.get("frequency", 1),
            "app_version": crash_data.get("app_version"),
            "os_version": crash_data.get("os_version")
        }
        
        return processed
    
    @trace_agent_step(name="analyze_patterns", metadata={"analysis_type": "pattern_recognition"})
    def _analyze_patterns(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in crash data"""
        # Simulate pattern analysis with LLM call
        patterns = self._call_llm_for_patterns(processed_data)
        
        return {
            "common_patterns": patterns,
            "device_patterns": {"high_frequency_devices": ["iPhone 12", "Pixel 5"]},
            "temporal_patterns": {"peak_hours": [14, 15, 16]}
        }
    
    @trace_llm_call(model="gpt-4o-mini", name="pattern_analysis")
    def _call_llm_for_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call LLM for pattern analysis"""
        # Simulate LLM call
        time.sleep(0.5)
        
        prompt = f"""
        Analyze the following crash data for patterns:
        
        Stack Trace: {data.get('clean_stack_trace', '')[:200]}...
        Device: {data.get('device_category')}
        Frequency: {data.get('crash_frequency')}
        
        Identify common patterns and categorize them.
        """
        
        # In real implementation, you would call the actual LLM
        # result = self.llm_client.complete(prompt)
        
        # Simulated result
        return {
            "memory_related": True,
            "ui_thread_crash": False,
            "network_related": False,
            "confidence": 0.85
        }
    
    @trace_agent_step(name="identify_root_causes")
    def _identify_root_causes(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Identify root causes based on patterns"""
        time.sleep(0.3)
        
        return {
            "primary_cause": "Memory leak in image processing",
            "secondary_causes": ["Inefficient bitmap handling", "Missing memory cleanup"],
            "confidence": 0.78
        }
    
    @trace_agent_step(name="generate_recommendations")
    def _generate_recommendations(self, root_causes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations"""
        time.sleep(0.2)
        
        return {
            "immediate_actions": [
                "Implement proper bitmap recycling",
                "Add memory pressure monitoring"
            ],
            "long_term_solutions": [
                "Refactor image processing pipeline",
                "Implement advanced memory management"
            ],
            "priority": "high"
        }
    
    @trace_agent_step(name="evaluate_quality")
    def _evaluate_analysis_quality(self, recommendations: Dict[str, Any]) -> float:
        """Evaluate the quality of the analysis"""
        time.sleep(0.1)
        
        # Simulate quality evaluation
        score = 0.82
        return score
    
    def _categorize_device(self, device_info: Dict[str, Any]) -> str:
        """Categorize device based on info"""
        # Simple device categorization logic
        if "iPhone" in device_info.get("model", ""):
            return "iOS_Premium"
        elif "Pixel" in device_info.get("model", ""):
            return "Android_Premium"
        else:
            return "Android_Standard"


def main():
    """Main example function"""
    print("🔍 Mobile Crash RCA Agent - Langfuse Integration Example")
    print("=" * 60)
    
    # Initialize agent
    agent = ObservableMobileCrashAgent()
    
    # Example crash data
    crash_data = {
        "crash_id": "crash_001",
        "user_id": "user_123",
        "app_version": "2.1.4",
        "os_version": "iOS 15.2",
        "device_info": {"model": "iPhone 12 Pro", "memory": "6GB"},
        "stack_trace": """
        Exception in thread "main" java.lang.OutOfMemoryError: Java heap space
        at com.app.ImageProcessor.loadBitmap(ImageProcessor.java:45)
        at com.app.MainActivity.onResume(MainActivity.java:123)
        """,
        "crash_type": "OutOfMemoryError",
        "frequency": 15
    }
    
    print(f"📱 Analyzing crash: {crash_data['crash_id']}")
    print(f"🔧 App version: {crash_data['app_version']}")
    print(f"📊 Frequency: {crash_data['frequency']} occurrences")
    print()
    
    # Run analysis
    start_time = time.time()
    result = agent.analyze_crash_data(crash_data)
    end_time = time.time()
    
    print("📈 Analysis Results:")
    print("=" * 20)
    
    if result["success"]:
        print(f"✅ Analysis completed successfully!")
        print(f"⚡ Processing time: {round((end_time - start_time) * 1000)}ms")
        print(f"🎯 Quality score: {result['quality_score']:.2f}")
        print(f"🔍 Session ID: {result['metadata']['session_id']}")
        print()
        
        print("🎯 Root Causes:")
        print(f"  Primary: {result['root_causes']['primary_cause']}")
        print(f"  Confidence: {result['root_causes']['confidence']:.2f}")
        print()
        
        print("💡 Recommendations:")
        for action in result['recommendations']['immediate_actions']:
            print(f"  • {action}")
        print()
        
        print("🌐 View detailed traces at: http://localhost:3000")
        
    else:
        print(f"❌ Analysis failed: {result['error']}")
    
    # Flush observability data
    agent.obs_handler.flush()
    print("📤 Observability data flushed to Langfuse")


if __name__ == "__main__":
    main() 