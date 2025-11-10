"""
Unit tests for the Reviewedness metric.
"""
import unittest
from unittest.mock import MagicMock, patch, Mock
from src.Metrics import ReviewednessMetric


class TestReviewednessMetric(unittest.TestCase):
    """Test cases for Reviewedness metric."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the git client to avoid API requirements
        with patch('src.Metrics.GitClient'):
            self.metric = ReviewednessMetric()
            self.metric.git_client = MagicMock()
    
    def test_extract_github_url_direct(self):
        """Test extraction when URL is already GitHub."""
        url = "https://github.com/owner/repo"
        
        result = self.metric._extract_github_url(url)
        
        self.assertEqual(result, url)
    
    def test_extract_github_url_not_github(self):
        """Test extraction when URL is not GitHub."""
        url = "https://huggingface.co/model/name"
        
        result = self.metric._extract_github_url(url)
        
        self.assertIsNone(result)
    
    def test_is_code_file_python(self):
        """Test code file detection for Python files."""
        self.assertTrue(self.metric._is_code_file("src/model.py"))
        self.assertTrue(self.metric._is_code_file("train.py"))
    
    def test_is_code_file_javascript(self):
        """Test code file detection for JavaScript files."""
        self.assertTrue(self.metric._is_code_file("app.js"))
        self.assertTrue(self.metric._is_code_file("utils.ts"))
    
    def test_is_code_file_weights(self):
        """Test that weight files are excluded."""
        self.assertFalse(self.metric._is_code_file("model.bin"))
        self.assertFalse(self.metric._is_code_file("weights.pt"))
        self.assertFalse(self.metric._is_code_file("checkpoint.pth"))
        self.assertFalse(self.metric._is_code_file("model.h5"))
    
    def test_is_code_file_data(self):
        """Test that data files are excluded."""
        self.assertFalse(self.metric._is_code_file("data.json"))
        self.assertFalse(self.metric._is_code_file("dataset.csv"))
        self.assertFalse(self.metric._is_code_file("data/train.txt"))
    
    def test_is_code_file_docs(self):
        """Test that documentation files are excluded."""
        self.assertFalse(self.metric._is_code_file("README.md"))
        self.assertFalse(self.metric._is_code_file("docs/guide.md"))
    
    def test_has_code_review_approved(self):
        """Test detection of approved reviews."""
        reviews = [
            {"state": "APPROVED", "user": {"login": "reviewer1"}}
        ]
        
        result = self.metric._has_code_review(reviews)
        
        self.assertTrue(result)
    
    def test_has_code_review_changes_requested(self):
        """Test detection of change requests."""
        reviews = [
            {"state": "CHANGES_REQUESTED", "user": {"login": "reviewer1"}}
        ]
        
        result = self.metric._has_code_review(reviews)
        
        self.assertTrue(result)
    
    def test_has_code_review_commented_only(self):
        """Test that comments without review state don't count."""
        reviews = [
            {"state": "COMMENTED", "user": {"login": "user1"}}
        ]
        
        result = self.metric._has_code_review(reviews)
        
        self.assertFalse(result)
    
    def test_has_code_review_empty(self):
        """Test with no reviews."""
        reviews = []
        
        result = self.metric._has_code_review(reviews)
        
        self.assertFalse(result)
    
    def test_get_pull_requests_success(self):
        """Test fetching pull requests successfully."""
        # Mock API response - return data on first call, empty on second
        call_count = [0]
        def mock_get(*args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            mock_response.status_code = 200
            if call_count[0] == 1:
                mock_response.json.return_value = [
                    {"number": 1, "state": "closed", "merged_at": "2024-01-01"},
                    {"number": 2, "state": "closed", "merged_at": "2024-01-02"}
                ]
            else:
                mock_response.json.return_value = []
            return mock_response
        
        self.metric.git_client.get = mock_get
        
        prs = self.metric._get_pull_requests("owner", "repo")
        
        self.assertEqual(len(prs), 2)
        self.assertEqual(prs[0]["number"], 1)
    
    def test_get_pull_requests_api_error(self):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        self.metric.git_client.get.return_value = mock_response
        
        prs = self.metric._get_pull_requests("owner", "repo")
        
        self.assertEqual(len(prs), 0)
    
    def test_get_pr_reviews_success(self):
        """Test fetching PR reviews."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"state": "APPROVED", "user": {"login": "reviewer"}}
        ]
        self.metric.git_client.get.return_value = mock_response
        
        reviews = self.metric._get_pr_reviews("owner", "repo", 1)
        
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["state"], "APPROVED")
    
    def test_get_pr_commits_success(self):
        """Test fetching PR commits."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"sha": "abc123"},
            {"sha": "def456"}
        ]
        self.metric.git_client.get.return_value = mock_response
        
        commits = self.metric._get_pr_commits("owner", "repo", 1)
        
        self.assertEqual(len(commits), 2)
        self.assertIn("abc123", commits)
    
    def test_get_commit_stats_with_code_files(self):
        """Test getting commit stats filtering for code files."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stats": {"additions": 100, "deletions": 50},
            "files": [
                {"filename": "src/model.py", "additions": 50, "deletions": 10},
                {"filename": "weights.bin", "additions": 40, "deletions": 30},
                {"filename": "app/utils.js", "additions": 10, "deletions": 10}
            ]
        }
        self.metric.git_client.get.return_value = mock_response
        
        stats = self.metric._get_commit_stats("owner", "repo", "abc123")
        
        # Should only count .py and .js files (60 additions + 20 deletions = 80 total)
        self.assertEqual(stats["additions"], 60)
        self.assertEqual(stats["deletions"], 20)
        self.assertEqual(stats["total"], 80)
    
    def test_compute_no_github_url(self):
        """Test compute with no GitHub URL."""
        inputs = {"model_url": "https://huggingface.co/model"}
        
        score = self.metric.compute(inputs)
        
        self.assertEqual(score, -1.0)
    
    def test_compute_invalid_github_url(self):
        """Test compute with invalid GitHub URL format."""
        inputs = {"git_url": "https://github.com/invalid"}
        
        score = self.metric.compute(inputs)
        
        self.assertEqual(score, -1.0)
    
    def test_compute_no_pull_requests(self):
        """Test compute when repo has no PRs."""
        inputs = {"git_url": "https://github.com/owner/repo"}
        
        # Mock empty PR list
        self.metric._get_pull_requests = MagicMock(return_value=[])
        
        score = self.metric.compute(inputs)
        
        self.assertEqual(score, 0.0)
    
    def test_compute_all_reviewed(self):
        """Test compute when all PRs are reviewed."""
        inputs = {"git_url": "https://github.com/owner/repo"}
        
        # Mock PRs
        mock_prs = [
            {"number": 1, "state": "closed", "merged_at": "2024-01-01"}
        ]
        self.metric._get_pull_requests = MagicMock(return_value=mock_prs)
        
        # Mock PR commits
        self.metric._get_pr_commits = MagicMock(return_value=["abc123"])
        
        # Mock reviews (approved)
        self.metric._get_pr_reviews = MagicMock(return_value=[
            {"state": "APPROVED", "user": {"login": "reviewer"}}
        ])
        
        # Mock commit stats
        self.metric._get_commit_stats = MagicMock(return_value={
            "additions": 100,
            "deletions": 50,
            "total": 150
        })
        
        score = self.metric.compute(inputs)
        
        # All commits reviewed = 1.0
        self.assertEqual(score, 1.0)
    
    def test_compute_partial_reviewed(self):
        """Test compute when some PRs are reviewed."""
        inputs = {"git_url": "https://github.com/owner/repo"}
        
        # Mock PRs
        mock_prs = [
            {"number": 1, "state": "closed", "merged_at": "2024-01-01"},
            {"number": 2, "state": "closed", "merged_at": "2024-01-02"}
        ]
        self.metric._get_pull_requests = MagicMock(return_value=mock_prs)
        
        # Mock PR commits
        def mock_get_pr_commits(owner, repo, pr_number):
            if pr_number == 1:
                return ["commit1"]
            else:
                return ["commit2"]
        self.metric._get_pr_commits = mock_get_pr_commits
        
        # Mock reviews (only PR 1 has review)
        def mock_get_pr_reviews(owner, repo, pr_number):
            if pr_number == 1:
                return [{"state": "APPROVED"}]
            else:
                return []
        self.metric._get_pr_reviews = mock_get_pr_reviews
        
        # Mock commit stats (equal lines for both)
        self.metric._get_commit_stats = MagicMock(return_value={
            "additions": 50,
            "deletions": 25,
            "total": 75
        })
        
        score = self.metric.compute(inputs)
        
        # 50% reviewed (1 of 2 PRs)
        self.assertEqual(score, 0.5)
    
    def test_compute_no_reviewed(self):
        """Test compute when no PRs have reviews."""
        inputs = {"git_url": "https://github.com/owner/repo"}
        
        # Mock PRs without reviews
        mock_prs = [
            {"number": 1, "state": "closed", "merged_at": "2024-01-01"}
        ]
        self.metric._get_pull_requests = MagicMock(return_value=mock_prs)
        self.metric._get_pr_commits = MagicMock(return_value=["abc123"])
        self.metric._get_pr_reviews = MagicMock(return_value=[])  # No reviews
        self.metric._get_commit_stats = MagicMock(return_value={
            "additions": 100,
            "deletions": 50,
            "total": 150
        })
        
        score = self.metric.compute(inputs)
        
        # No reviews = 0.0
        self.assertEqual(score, 0.0)
    
    def test_compute_unmerged_prs_ignored(self):
        """Test that unmerged PRs are not counted."""
        inputs = {"git_url": "https://github.com/owner/repo"}
        
        # Mock PR without merged_at
        mock_prs = [
            {"number": 1, "state": "open", "merged_at": None}
        ]
        self.metric._get_pull_requests = MagicMock(return_value=mock_prs)
        
        score = self.metric.compute(inputs)
        
        # No merged PRs = 0.0
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
