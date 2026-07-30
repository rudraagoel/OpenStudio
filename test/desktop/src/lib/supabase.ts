import { createClient } from '@supabase/supabase-js';

// TODO: Replace these with real credentials from the .env file once provided by the user
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://placeholder.supabase.co';
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'placeholder_key';

export const supabase = createClient(supabaseUrl, supabaseKey);

// Interfaces for our P2P and Jobs schema
export interface ComputeNode {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'busy';
  gpu_type: string;
  vram_available: number;
  last_heartbeat: string;
}

export interface Job {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  parameters: any;
  result_url?: string;
  eta_seconds?: number;
  assigned_node?: string;
}
