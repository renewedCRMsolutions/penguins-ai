# File: scrape_nhl_repos.py
# Scrape and analyze NHL GitHub repositories for data and methods

import requests
import json
import base64
# from datetime import datetime  # Unused import
import os
import time

class NHLRepoScraper:
    """Scrape NHL repositories for data sources and methods"""
    
    def __init__(self):
        self.repos = [
            {
                'owner': 'cole-titze',
                'name': 'nhl-odds',
                'description': 'NHL odds and betting data'
            },
            {
                'owner': 'okkonie',
                'name': 'nhl-api-endpoints',
                'description': 'NHL API endpoint documentation'
            },
            {
                'owner': 'Zmalski',
                'name': 'NHL-API-Reference',
                'description': 'Comprehensive NHL API reference'
            }
        ]
        self.github_api = "https://api.github.com"
        self.findings = {
            'data_sources': [],
            'api_endpoints': [],
            'data_models': [],
            'processing_methods': []
        }
    
    def scrape_all_repos(self):
        """Scrape all NHL repositories"""
        print("🔍 Scraping NHL GitHub Repositories...")
        print("=" * 60)
        
        for repo in self.repos:
            print(f"\n📂 Analyzing: {repo['owner']}/{repo['name']}")
            print(f"   Description: {repo['description']}")
            self.scrape_repo(repo['owner'], repo['name'])
            time.sleep(1)  # Rate limiting
        
        self.save_findings()
        self.generate_recommendations()
    
    def scrape_repo(self, owner, repo_name):
        """Scrape a single repository"""
        # Get repository structure
        url = f"{self.github_api}/repos/{owner}/{repo_name}/contents"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                contents = response.json()
                self.analyze_repo_contents(owner, repo_name, contents)
            else:
                print(f"   ❌ Error accessing repo: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    def analyze_repo_contents(self, owner, repo_name, contents, path=""):
        """Recursively analyze repository contents"""
        for item in contents:
            if item['type'] == 'file':
                # Look for interesting files
                if any(keyword in item['name'].lower() for keyword in 
                       ['api', 'endpoint', 'data', 'model', 'fetch', 'scrape']):
                    print(f"   📄 Found: {path}/{item['name']}")
                    self.analyze_file(owner, repo_name, item)
                    
            elif item['type'] == 'dir':
                # Recursively explore directories
                if item['name'] not in ['node_modules', '.git', '__pycache__']:
                    subdir_url = item['url']
                    try:
                        response = requests.get(subdir_url)
                        if response.status_code == 200:
                            subdir_contents = response.json()
                            self.analyze_repo_contents(
                                owner, repo_name, subdir_contents, 
                                f"{path}/{item['name']}"
                            )
                    except:
                        pass
                    time.sleep(0.5)  # Rate limiting
    
    def analyze_file(self, owner, repo_name, file_info):
        """Analyze a specific file for useful information"""
        try:
            # Get file content
            response = requests.get(file_info['url'])
            if response.status_code == 200:
                data = response.json()
                content = base64.b64decode(data['content']).decode('utf-8')
                
                # Analyze based on file type
                if file_info['name'].endswith(('.py', '.js', '.ts')):
                    self.extract_code_insights(content, file_info['name'], repo_name)
                elif file_info['name'].endswith(('.json', '.yaml', '.yml')):
                    self.extract_config_insights(content, file_info['name'], repo_name)
                elif file_info['name'].endswith(('.md', '.txt')):
                    self.extract_documentation_insights(content, file_info['name'], repo_name)
                    
        except Exception as e:
            print(f"      ⚠️ Could not analyze {file_info['name']}: {str(e)}")
    
    def extract_code_insights(self, content, filename, repo_name):
        """Extract insights from code files"""
        insights = []
        
        # Look for API endpoints
        if 'api-web.nhle.com' in content:
            print(f"      ✓ Found NHL API usage")
            endpoints = self.extract_endpoints(content)
            for endpoint in endpoints:
                self.findings['api_endpoints'].append({
                    'endpoint': endpoint,
                    'source': f"{repo_name}/{filename}",
                    'type': 'official_nhl_api'
                })
        
        # Look for data processing methods
        if any(keyword in content.lower() for keyword in 
               ['def process', 'function process', 'def extract', 'def transform']):
            print(f"      ✓ Found data processing methods")
            methods = self.extract_methods(content)
            self.findings['processing_methods'].extend(methods)
        
        # Look for data models
        if any(keyword in content for keyword in 
               ['class', 'interface', 'schema', 'model']):
            models = self.extract_models(content)
            if models:
                print(f"      ✓ Found {len(models)} data models")
                self.findings['data_models'].extend(models)
    
    def extract_endpoints(self, content):
        """Extract API endpoints from code"""
        import re
        endpoints = []
        
        # Pattern for NHL API endpoints
        patterns = [
            r'["\']/([\w\-/{}]+)["\']',  # General endpoints
            r'https://api-web\.nhle\.com/v1/([\w\-/{}]+)',  # Full URLs
            r'endpoint["\']?\s*[:=]\s*["\']/([\w\-/{}]+)["\']',  # Named endpoints
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            endpoints.extend(matches)
        
        # Clean and deduplicate
        cleaned = []
        for endpoint in endpoints:
            if endpoint and not any(skip in endpoint for skip in ['node_modules', 'test']):
                cleaned.append(endpoint)
        
        return list(set(cleaned))
    
    def extract_methods(self, content):
        """Extract data processing methods"""
        import re
        methods = []
        
        # Python methods
        python_methods = re.findall(r'def (\w+)\(.*?\):\s*"""(.*?)"""', content, re.DOTALL)
        for name, docstring in python_methods:
            if any(keyword in name.lower() for keyword in ['fetch', 'process', 'extract', 'transform']):
                methods.append({
                    'name': name,
                    'description': docstring.strip(),
                    'language': 'python'
                })
        
        # JavaScript/TypeScript methods
        js_methods = re.findall(r'(?:function|const)\s+(\w+).*?{([^}]+)}', content, re.DOTALL)
        for name, body in js_methods:
            if any(keyword in name.lower() for keyword in ['fetch', 'process', 'extract', 'transform']):
                methods.append({
                    'name': name,
                    'language': 'javascript'
                })
        
        return methods
    
    def extract_models(self, content):
        """Extract data model definitions"""
        import re
        models = []
        
        # Python classes
        python_classes = re.findall(r'class (\w+).*?:\s*"""(.*?)"""', content, re.DOTALL)
        for name, docstring in python_classes:
            if any(keyword in name.lower() for keyword in ['shot', 'game', 'player', 'team']):
                models.append({
                    'name': name,
                    'description': docstring.strip(),
                    'type': 'python_class'
                })
        
        # TypeScript interfaces
        ts_interfaces = re.findall(r'interface (\w+)\s*{([^}]+)}', content, re.DOTALL)
        for name, body in ts_interfaces:
            if any(keyword in name.lower() for keyword in ['shot', 'game', 'player', 'team']):
                fields = re.findall(r'(\w+)\s*:\s*([\w\[\]]+)', body)
                models.append({
                    'name': name,
                    'fields': fields,
                    'type': 'typescript_interface'
                })
        
        return models
    
    def extract_config_insights(self, content, filename, repo_name):
        """Extract insights from configuration files"""
        try:
            data = json.loads(content)
            
            # Look for API configurations
            if 'api' in str(data).lower():
                self.findings['data_sources'].append({
                    'type': 'api_config',
                    'source': f"{repo_name}/{filename}",
                    'content': data
                })
                print(f"      ✓ Found API configuration")
        except:
            pass  # Not JSON or parsing error
    
    def extract_documentation_insights(self, content, filename, repo_name):
        """Extract insights from documentation"""
        if 'endpoint' in content.lower() or 'api' in content.lower():
            # Extract documented endpoints
            import re
            endpoint_patterns = [
                r'`GET\s+([^`]+)`',
                r'`POST\s+([^`]+)`',
                r'endpoint:\s*([^\n]+)',
                r'/v1/([^\s\n]+)',
            ]
            
            for pattern in endpoint_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    self.findings['api_endpoints'].append({
                        'endpoint': match.strip(),
                        'source': f"{repo_name}/{filename}",
                        'type': 'documented'
                    })
            
            if matches:
                print(f"      ✓ Found {len(matches)} documented endpoints")
    
    def save_findings(self):
        """Save all findings to a file"""
        output_file = 'data/nhl_repo_analysis.json'
        os.makedirs('data', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.findings, f, indent=2)
        
        print(f"\n💾 Findings saved to {output_file}")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   - API Endpoints found: {len(self.findings['api_endpoints'])}")
        print(f"   - Data models found: {len(self.findings['data_models'])}")
        print(f"   - Processing methods found: {len(self.findings['processing_methods'])}")
        print(f"   - Data sources found: {len(self.findings['data_sources'])}")
    
    def generate_recommendations(self):
        """Generate recommendations based on findings"""
        print("\n🎯 Recommendations based on repository analysis:")
        print("=" * 60)
        
        # Unique endpoints
        unique_endpoints = set()
        for ep in self.findings['api_endpoints']:
            unique_endpoints.add(ep['endpoint'])
        
        print("\n1. Key NHL API Endpoints to implement:")
        for i, endpoint in enumerate(list(unique_endpoints)[:10], 1):
            print(f"   {i}. /v1/{endpoint}")
        
        print("\n2. Data models to create:")
        for model in self.findings['data_models'][:5]:
            print(f"   - {model['name']}: {model.get('description', 'No description')[:60]}...")
        
        print("\n3. Processing methods to implement:")
        for method in self.findings['processing_methods'][:5]:
            print(f"   - {method['name']} ({method['language']})")
        
        print("\n4. Next steps:")
        print("   - Review data/nhl_repo_analysis.json for complete findings")
        print("   - Implement the discovered endpoints in your training script")
        print("   - Use the data models found in the repos")
        print("   - Apply the processing methods to your pipeline")


def main():
    """Run the repository scraper"""
    scraper = NHLRepoScraper()
    scraper.scrape_all_repos()
    
    print("\n✅ Repository analysis complete!")
    print("   Check data/nhl_repo_analysis.json for detailed findings")


if __name__ == "__main__":
    main()