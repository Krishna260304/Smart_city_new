import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';

const DemoLogin = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleDemoLogin = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: 'raj@example.com',
          password: 'password123',
        }),
      });

      const data = await response.json();

      if (data.success && data.data.token) {
        localStorage.setItem('auth_token', data.data.token);
        localStorage.setItem('user', JSON.stringify(data.data.user));

        navigate('/dashboard');
      } else {
        setError(data.error || 'Login failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <Card className="w-full max-w-md shadow-xl">
        <CardHeader className="space-y-2 text-center">
          <CardTitle className="text-2xl">SafeLive Demo</CardTitle>
          <CardDescription>Quick login to view the dashboard with sample incidents</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-2">
            <p className="font-medium text-sm text-blue-900">Demo Credentials</p>
            <p className="text-sm text-blue-700">
              Email: <code className="bg-white px-2 py-1 rounded">raj@example.com</code>
            </p>
            <p className="text-sm text-blue-700">
              Password: <code className="bg-white px-2 py-1 rounded">password123</code>
            </p>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          <Button
            onClick={handleDemoLogin}
            disabled={loading}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Logging in...
              </>
            ) : (
              'Login & View Dashboard'
            )}
          </Button>

          <p className="text-xs text-gray-500 text-center">
            The database has been seeded with 8 sample incidents and multiple users.
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default DemoLogin;
