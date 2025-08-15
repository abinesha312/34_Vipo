import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Folder, RefreshCw, Upload, FileText, Trash2 } from "lucide-react";

interface Stats {
  total_documents: number;
  pdf_files: number;
  txt_files: number;
  vector_db_size_mb: number;
  total_chunks: number;
  last_updated: string;
}

interface Document {
  name: string;
  type: string;
  size_mb: number;
  modified: string;
}

const VipoDashboard = () => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch statistics",
        variant: "destructive",
      });
    }
  };

  const fetchDocuments = async () => {
    try {
      const response = await fetch('/api/documents');
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch documents",
        variant: "destructive",
      });
    }
  };

  const handleFileUpload = async (files: FileList) => {
    if (!files.length) return;

    setLoading(true);
    const formData = new FormData();
    
    for (const file of files) {
      formData.append('files', file);
    }

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (response.ok) {
        toast({
          title: "Success",
          description: result.message,
        });
        await fetchStats();
        await fetchDocuments();
      } else {
        throw new Error(result.error || 'Upload failed');
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Upload failed',
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDocument = async (filename: string) => {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) return;

    try {
      const response = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      });

      const result = await response.json();
      
      if (response.ok) {
        toast({
          title: "Success",
          description: result.message,
        });
        await fetchStats();
        await fetchDocuments();
      } else {
        throw new Error(result.error || 'Delete failed');
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Delete failed',
        variant: "destructive",
      });
    }
  };

  const handleReprocess = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/reprocess', {
        method: 'POST',
      });

      const result = await response.json();
      
      if (response.ok) {
        toast({
          title: "Success",
          description: result.message,
        });
        await fetchStats();
      } else {
        throw new Error(result.error || 'Reprocess failed');
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Reprocess failed',
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  useEffect(() => {
    fetchStats();
    fetchDocuments();
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-vipo-blue text-vipo-blue-foreground py-8 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <div className="flex items-center justify-center gap-3 mb-2">
            <div className="text-3xl">🏛️</div>
            <h1 className="text-4xl font-bold">Vipo Knowledge Base</h1>
          </div>
          <p className="text-xl opacity-90">Banking Policy Document Management & RAG System</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6 space-y-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="bg-vipo-amber text-vipo-amber-foreground p-6 border-2 border-black rounded-lg">
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">{stats?.total_documents || 0}</div>
              <div className="text-sm font-medium">Total Documents</div>
            </div>
          </Card>
          
          <Card className="bg-vipo-amber text-vipo-amber-foreground p-6 border-2 border-black rounded-lg">
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">{stats?.total_chunks || 0}</div>
              <div className="text-sm font-medium">Total Chunks</div>
            </div>
          </Card>
          
          <Card className="bg-vipo-amber text-vipo-amber-foreground p-6 border-2 border-black rounded-lg">
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">
                {stats?.vector_db_size_mb?.toFixed(2) || '0.00'}
              </div>
              <div className="text-sm font-medium">Vector DB (MB)</div>
            </div>
          </Card>
          
          <Card className="bg-vipo-amber text-vipo-amber-foreground p-6 border-2 border-black rounded-lg">
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">
                {stats?.last_updated ? formatDate(stats.last_updated) : 'N/A'}
              </div>
              <div className="text-sm font-medium">Last Updated</div>
            </div>
          </Card>
        </div>

        {/* Upload Section */}
        <Card className="border-2 border-dashed border-vipo-upload-border bg-vipo-upload-background p-8">
          <div className="text-center space-y-4">
            <div className="flex items-center justify-center gap-2 text-lg font-medium">
              <Folder className="w-5 h-5" />
              Upload Documents
            </div>
            
            <div>
              <input
                type="file"
                multiple
                accept=".pdf,.txt"
                onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
                className="hidden"
                id="file-upload"
                disabled={loading}
              />
              <label htmlFor="file-upload">
                <Button 
                  className="bg-vipo-blue hover:bg-vipo-blue/90 text-vipo-blue-foreground px-8 py-2 rounded-lg cursor-pointer"
                  disabled={loading}
                  asChild
                >
                  <span className="flex items-center gap-2">
                    <Upload className="w-4 h-4" />
                    Choose Files
                  </span>
                </Button>
              </label>
            </div>
          </div>
        </Card>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button 
            onClick={handleReprocess}
            disabled={loading}
            className="bg-vipo-amber hover:bg-vipo-amber/90 text-vipo-amber-foreground px-6 py-2 rounded-lg"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            🔄 Reprocess All Documents
          </Button>
          
          <Button 
            onClick={() => { fetchStats(); fetchDocuments(); }}
            disabled={loading}
            className="bg-vipo-amber hover:bg-vipo-amber/90 text-vipo-amber-foreground px-6 py-2 rounded-lg"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            📊 Refresh Stats
          </Button>
        </div>

        {/* Document Library */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-6">
            <FileText className="w-5 h-5" />
            <h2 className="text-lg font-semibold">📚 Document Library</h2>
          </div>
          
          <div className="space-y-4">
            {documents.length === 0 ? (
              <p className="text-muted-foreground text-center py-8">No documents uploaded yet</p>
            ) : (
              documents.map((doc, index) => (
                <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <div className="font-medium">{doc.name}</div>
                    <div className="text-sm text-muted-foreground">
                      {doc.type} • {doc.size_mb.toFixed(2)} MB • Modified: {formatDate(doc.modified)}
                    </div>
                  </div>
                  <Button
                    onClick={() => handleDeleteDocument(doc.name)}
                    variant="destructive"
                    size="sm"
                    className="flex items-center gap-2"
                  >
                    <Trash2 className="w-4 h-4" />
                    🗑️ Delete
                  </Button>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default VipoDashboard;