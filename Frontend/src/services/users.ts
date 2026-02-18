import { apiClient, ApiResponse } from './api';
import { API_ENDPOINTS } from '@/config/api';
import { OfficialRole } from './auth';

export interface UserProfile {
  id: string;
  name: string;
  email?: string;
  phone?: string;
  userType: 'citizen' | 'official';
  officialRole?: OfficialRole;
  workerSpecialization?: string;
  address?: string;
  pincode?: string;
  department?: string;
  twoFactorEnabled?: boolean;
}

export interface UserProfileUpdate {
  name?: string;
  email?: string;
  phone?: string;
  address?: string;
  pincode?: string;
  department?: string;
}

export interface WorkerOption {
  id: string;
  name: string;
  phone?: string;
  email?: string;
  officialRole?: OfficialRole;
  workerSpecialization?: string;
}

export const usersService = {
  async getProfile(): Promise<ApiResponse<UserProfile>> {
    return apiClient.get<UserProfile>(API_ENDPOINTS.USERS.PROFILE);
  },

  async updateProfile(payload: UserProfileUpdate): Promise<ApiResponse<UserProfile>> {
    return apiClient.put<UserProfile>(API_ENDPOINTS.USERS.UPDATE_PROFILE, payload);
  },

  async listWorkers(): Promise<ApiResponse<WorkerOption[]>> {
    return apiClient.get<WorkerOption[]>(API_ENDPOINTS.USERS.WORKERS);
  },
};
