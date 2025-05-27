# File: research/analyze_local_repos.py
# Analyze local NHL repositories for endpoints, data models, and techniques

import os
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import ast


class LocalRepoAnalyzer:
    """Analyze local NHL repositories for useful information"""

    def __init__(self):
        self.repos = {
            "NHL-API-Reference": "Comprehensive API documentation",
            "nhl-api-endpoints": "API endpoint implementations",
            "nhl-odds": "NHL betting and analytics",
        }
        self.findings = {
            "endpoints": defaultdict(list),
            "data_models": [],
            "features": [],
            "processing_methods": [],
            "useful_code": [],
            "database_schemas": [],
        }
        self.file_count = 0
        self.insights = []

    def analyze_all_repos(self):
        """Main analysis function"""
        print("🔬 NHL REPOSITORY ANALYSIS TOOL")
        print("=" * 70)
        print("Analyzing local repositories for NHL data insights...\n")

        for repo_name, description in self.repos.items():
            if os.path.exists(repo_name):
                print(f"\n📁 Analyzing: {repo_name}")
                print(f"   Description: {description}")
                self.analyze_repo(repo_name)
            else:
                print(f"\n❌ Repository not found: {repo_name}")
                print(f"   Make sure it's in the current directory")

        self.generate_report()

    def analyze_repo(self, repo_path):
        """Analyze a single repository"""
        for root, dirs, files in os.walk(repo_path):
            # Skip common non-useful directories
            dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "__pycache__", ".pytest_cache"]]

            for file in files:
                file_path = os.path.join(root, file)
                self.file_count += 1

                # Analyze based on file type
                if file.endswith(".py"):
                    self.analyze_python_file(file_path, repo_path)
                elif file.endswith((".js", ".ts")):
                    self.analyze_javascript_file(file_path, repo_path)
                elif file.endswith(".json"):
                    self.analyze_json_file(file_path, repo_path)
                elif file.endswith(".sql"):
                    self.analyze_sql_file(file_path, repo_path)
                elif file.endswith((".md", ".txt")) and "readme" in file.lower():
                    self.analyze_documentation(file_path, repo_path)

    def analyze_python_file(self, file_path, repo_name):
        """Extract insights from Python files"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find API endpoints
            endpoints = self.extract_endpoints(content)
            for endpoint in endpoints:
                self.findings["endpoints"][endpoint].append({"file": file_path, "repo": repo_name})

            # Find data processing functions
            functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):\s*"""([^"]*?)"""', content, re.DOTALL)
            for func_name, docstring in functions:
                if any(keyword in func_name.lower() for keyword in ["process", "extract", "calculate", "fetch", "get"]):
                    self.findings["processing_methods"].append(
                        {"function": func_name, "description": docstring.strip(), "file": file_path, "repo": repo_name}
                    )

            # Find feature calculations
            if "feature" in content.lower() or "calculate" in content.lower():
                features = self.extract_features(content)
                self.findings["features"].extend(features)

            # Look for shot-related code
            if "shot" in content.lower():
                shot_code = self.extract_shot_logic(content)
                if shot_code:
                    self.findings["useful_code"].append(
                        {"type": "shot_processing", "file": file_path, "code_snippet": shot_code}
                    )

        except Exception as e:
            pass  # Skip files that can't be read

    def analyze_javascript_file(self, file_path, repo_name):
        """Extract insights from JavaScript/TypeScript files"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find API endpoints
            endpoints = self.extract_endpoints(content)
            for endpoint in endpoints:
                self.findings["endpoints"][endpoint].append({"file": file_path, "repo": repo_name})

            # Find interfaces/types (data models)
            interfaces = re.findall(r"interface\s+(\w+)\s*{([^}]+)}", content)
            for interface_name, interface_body in interfaces:
                if any(keyword in interface_name.lower() for keyword in ["shot", "game", "player", "stat"]):
                    self.findings["data_models"].append(
                        {
                            "name": interface_name,
                            "type": "TypeScript Interface",
                            "file": file_path,
                            "fields": self.extract_interface_fields(interface_body),
                        }
                    )

        except Exception as e:
            pass

    def analyze_json_file(self, file_path, repo_name):
        """Analyze JSON files for API documentation or data"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if it's API documentation
            if isinstance(data, dict):
                if "endpoints" in data or "api" in str(data).lower():
                    self.findings["useful_code"].append(
                        {
                            "type": "api_documentation",
                            "file": file_path,
                            "summary": f"Found {len(data)} API definitions",
                        }
                    )

                # Look for example responses with shot data
                if "shot" in str(data).lower():
                    self.analyze_shot_data_structure(data, file_path)

        except Exception as e:
            pass

    def analyze_sql_file(self, file_path, repo_name):
        """Extract database schemas from SQL files"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find CREATE TABLE statements
            tables = re.findall(r"CREATE TABLE\s+(\w+)\s*\((.*?)\);", content, re.DOTALL | re.IGNORECASE)
            for table_name, table_def in tables:
                columns = self.extract_sql_columns(table_def)
                self.findings["database_schemas"].append(
                    {"table": table_name, "columns": columns, "file": file_path, "repo": repo_name}
                )

                # Special interest in shot/game related tables
                if any(keyword in table_name.lower() for keyword in ["shot", "game", "event", "play"]):
                    self.insights.append(f"Found {table_name} table with shot/game data structure")

        except Exception as e:
            pass

    def analyze_documentation(self, file_path, repo_name):
        """Extract information from README and documentation files"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for endpoint documentation
            endpoint_sections = re.findall(r"#+\s*Endpoint[s]?\s*\n(.*?)(?=\n#|\Z)", content, re.DOTALL)
            for section in endpoint_sections:
                documented_endpoints = re.findall(r"`([/\w\-{}]+)`", section)
                for endpoint in documented_endpoints:
                    self.findings["endpoints"][endpoint].append(
                        {"file": file_path, "repo": repo_name, "documented": True}
                    )

            # Extract feature lists
            if "feature" in content.lower():
                features = re.findall(r"[-*]\s*([^\n]+feature[^\n]+)", content, re.IGNORECASE)
                self.findings["features"].extend(features[:5])  # Top 5 features mentioned

        except Exception as e:
            pass

    def extract_endpoints(self, content):
        """Extract API endpoints from code"""
        endpoints = []

        # Common patterns for API endpoints
        patterns = [
            r'["\']/(v\d+/)?([a-zA-Z0-9\-/{}]+)["\']',  # General endpoints
            r"api-web\.nhle\.com/v1/([a-zA-Z0-9\-/{}]+)",  # NHL specific
            r'endpoint["\']?\s*[:=]\s*["\']/([\w\-/{}]+)["\']',  # Named endpoints
            r'url\s*=.*?["\']/([\w\-/{}]+)["\']',  # URL assignments
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                endpoint = match if isinstance(match, str) else match[-1]
                # Clean up and validate
                if (
                    endpoint
                    and len(endpoint) > 3
                    and not any(skip in endpoint for skip in ["node_modules", "test", ".js", ".css"])
                ):
                    endpoints.append(endpoint)

        return list(set(endpoints))  # Remove duplicates

    def extract_features(self, content):
        """Extract feature calculations from code"""
        features = []

        # Look for feature-related patterns
        feature_patterns = [
            r"(?:feature|calculate|compute)_(\w+)",
            r"(\w+)_feature",
            r"def calculate_(\w+)",
        ]

        for pattern in feature_patterns:
            matches = re.findall(pattern, content)
            features.extend(matches)

        # Look for specific hockey features
        hockey_features = re.findall(r"(shot_\w+|xg_\w+|expected_\w+|danger_\w+)", content, re.IGNORECASE)
        features.extend(hockey_features)

        return list(set(features))

    def extract_shot_logic(self, content):
        """Extract shot-related code logic"""
        # Find functions that process shots
        shot_functions = re.findall(r"def\s+\w*shot\w*\s*\([^)]*\):[^}]+?(?=\ndef|\Z)", content, re.DOTALL)
        if shot_functions:
            return shot_functions[0][:500]  # First 500 chars

        # Find shot calculations
        shot_calcs = re.findall(r"(shot.*?=.*?\n(?:.*?\n){0,5})", content)
        if shot_calcs:
            return shot_calcs[0]

        return None

    def extract_interface_fields(self, interface_body):
        """Extract fields from TypeScript interface"""
        fields = re.findall(r"(\w+)\s*:\s*([\w\[\]<>]+)", interface_body)
        return [f"{name}: {type_}" for name, type_ in fields]

    def extract_sql_columns(self, table_def):
        """Extract columns from SQL CREATE TABLE"""
        columns = []
        lines = table_def.strip().split("\n")
        for line in lines:
            match = re.match(r"\s*(\w+)\s+(\w+)", line)
            if match:
                columns.append(f"{match.group(1)} ({match.group(2)})")
        return columns

    def analyze_shot_data_structure(self, data, file_path):
        """Analyze JSON structure for shot data"""

        def find_shot_keys(obj, path=""):
            shot_keys = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if "shot" in key.lower():
                        shot_keys.append(f"{path}.{key}" if path else key)
                    if isinstance(value, (dict, list)):
                        shot_keys.extend(find_shot_keys(value, f"{path}.{key}" if path else key))
            elif isinstance(obj, list) and obj:
                shot_keys.extend(find_shot_keys(obj[0], f"{path}[0]"))
            return shot_keys

        shot_keys = find_shot_keys(data)
        if shot_keys:
            self.insights.append(f"Found shot data structure in {file_path}: {shot_keys[:3]}")

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 70)
        print("📊 ANALYSIS COMPLETE")
        print("=" * 70)

        # Summary statistics
        print(f"\n📈 Summary:")
        print(f"  Files analyzed: {self.file_count}")
        print(f"  Unique endpoints found: {len(self.findings['endpoints'])}")
        print(f"  Data models found: {len(self.findings['data_models'])}")
        print(f"  Processing methods: {len(self.findings['processing_methods'])}")
        print(f"  Database tables: {len(self.findings['database_schemas'])}")

        # Top endpoints
        print(f"\n🎯 Top API Endpoints:")
        for i, (endpoint, sources) in enumerate(list(self.findings["endpoints"].items())[:10], 1):
            print(f"  {i}. {endpoint}")
            print(f"     Found in: {sources[0]['repo']}")

        # Useful processing methods
        print(f"\n⚙️ Key Processing Methods Found:")
        for method in self.findings["processing_methods"][:5]:
            print(f"  - {method['function']}() in {method['repo']}")
            if method["description"]:
                print(f"    {method['description'][:60]}...")

        # Database insights
        if self.findings["database_schemas"]:
            print(f"\n🗄️ Database Tables Found:")
            for schema in self.findings["database_schemas"][:5]:
                print(f"  - {schema['table']} ({len(schema['columns'])} columns)")
                print(f"    Key columns: {', '.join(schema['columns'][:3])}")

        # Features discovered
        if self.findings["features"]:
            print(f"\n🔧 Features/Calculations Found:")
            unique_features = list(set(self.findings["features"]))
            for feature in unique_features[:10]:
                print(f"  - {feature}")

        # Save detailed report
        self.save_detailed_report()

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("1. Implement these endpoints for richer data:")
        priority_endpoints = [ep for ep in self.findings["endpoints"] if "shot" in ep or "play" in ep or "event" in ep]
        for ep in priority_endpoints[:5]:
            print(f"   - {ep}")

        print("\n2. Add these features to your model:")
        shot_features = [f for f in self.findings["features"] if "shot" in f.lower() or "xg" in f.lower()]
        for feature in shot_features[:5]:
            print(f"   - {feature}")

        print("\n3. Review the full report at: research/repo_analysis_report.json")

    def save_detailed_report(self):
        """Save detailed findings to JSON"""
        os.makedirs("research", exist_ok=True)

        report = {
            "analysis_date": datetime.now().isoformat(),
            "repositories_analyzed": list(self.repos.keys()),
            "files_analyzed": self.file_count,
            "findings": {
                "endpoints": dict(self.findings["endpoints"]),
                "data_models": self.findings["data_models"],
                "processing_methods": self.findings["processing_methods"],
                "database_schemas": self.findings["database_schemas"],
                "features": list(set(self.findings["features"])),
                "useful_code_snippets": len(self.findings["useful_code"]),
            },
            "insights": self.insights,
            "implementation_priority": self.generate_implementation_plan(),
        }

        with open("research/repo_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Also create a markdown summary
        self.create_markdown_summary(report)

    def generate_implementation_plan(self):
        """Generate prioritized implementation plan"""
        plan = {"immediate": [], "high_priority": [], "nice_to_have": []}

        # Immediate: Shot-related endpoints
        for endpoint in self.findings["endpoints"]:
            if any(keyword in endpoint.lower() for keyword in ["shot", "play-by-play", "event"]):
                plan["immediate"].append(f"Implement endpoint: /{endpoint}")

        # High priority: Processing methods
        for method in self.findings["processing_methods"][:5]:
            plan["high_priority"].append(f"Port {method['function']} from {method['repo']}")

        # Nice to have: Advanced features
        for feature in self.findings["features"][:5]:
            plan["nice_to_have"].append(f"Calculate {feature} feature")

        return plan

    def create_markdown_summary(self, report):
        """Create a readable markdown summary"""
        with open("research/REPO_ANALYSIS_SUMMARY.md", "w") as f:
            f.write("# NHL Repository Analysis Summary\n\n")
            f.write(f"Generated: {report['analysis_date']}\n\n")

            f.write("## Key Findings\n\n")
            f.write(f"- **Endpoints discovered**: {len(report['findings']['endpoints'])}\n")
            f.write(f"- **Processing methods**: {len(report['findings']['processing_methods'])}\n")
            f.write(f"- **Data models**: {len(report['findings']['data_models'])}\n")
            f.write(f"- **Database schemas**: {len(report['findings']['database_schemas'])}\n\n")

            f.write("## Top Endpoints to Implement\n\n")
            for i, endpoint in enumerate(list(report["findings"]["endpoints"].keys())[:10], 1):
                f.write(f"{i}. `/{endpoint}`\n")

            f.write("\n## Implementation Priority\n\n")
            f.write("### Immediate Actions\n")
            for action in report["implementation_priority"]["immediate"][:5]:
                f.write(f"- {action}\n")

            f.write("\n### High Priority\n")
            for action in report["implementation_priority"]["high_priority"][:5]:
                f.write(f"- {action}\n")


def main():
    """Run the repository analysis"""
    analyzer = LocalRepoAnalyzer()
    analyzer.analyze_all_repos()


if __name__ == "__main__":
    main()
