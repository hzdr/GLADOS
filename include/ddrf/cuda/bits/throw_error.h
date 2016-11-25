/*
 * This file is part of the ddrf library.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * ddrf is free software: You can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ddrf is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ddrf. If not, see <http://www.gnu.org/licenses/>.
 * 
 * Date: 15 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef DDRF_CUDA_BITS_THROW_ERROR_H_
#define DDRF_CUDA_BITS_THROW_ERROR_H_

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <ddrf/cuda/exception.h>

namespace ddrf
{
    namespace cuda
    {
        namespace detail
        {
            inline auto throw_error(cudaError_t err) -> void
            {
                // disable warnings for obsolete cases
                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Wswitch-enum"
                switch(err)
                {
                    case cudaSuccess:
                        throw std::runtime_error{"Mishandled cudaSuccess passed to error selector."};

                    case cudaErrorNotReady:
                        throw std::runtime_error{"Mishandled cudaErrorNotReady passed to error selector."};

                    case cudaErrorMissingConfiguration:
                    case cudaErrorInitializationError:
                    case cudaErrorLaunchFailure:
                    case cudaErrorLaunchTimeout:
                    case cudaErrorInvalidDeviceFunction:
                    case cudaErrorMapBufferObjectFailed:
                    case cudaErrorUnmapBufferObjectFailed:
                    case cudaErrorCudartUnloading:
                    case cudaErrorUnknown:
                    case cudaErrorInsufficientDriver:
                    case cudaErrorNoDevice:
                    case cudaErrorECCUncorrectable:
                    case cudaErrorSharedObjectSymbolNotFound:
                    case cudaErrorSharedObjectInitFailed:
                    case cudaErrorDevicesUnavailable:
                    case cudaErrorIncompatibleDriverContext:
                    case cudaErrorDeviceAlreadyInUse:
                    case cudaErrorProfilerDisabled:
                    case cudaErrorAssert:
                    case cudaErrorTooManyPeers:
                    case cudaErrorOperatingSystem:
                    case cudaErrorPeerAccessUnsupported:
                    case cudaErrorLaunchMaxDepthExceeded:
                    case cudaErrorSyncDepthExceeded:
                    case cudaErrorLaunchPendingCountExceeded:
                    case cudaErrorNotPermitted:
                    case cudaErrorNotSupported:
                    case cudaErrorHardwareStackError:
                    case cudaErrorIllegalInstruction:
                    case cudaErrorMisalignedAddress:
                    case cudaErrorInvalidAddressSpace:
                    case cudaErrorInvalidPc:
                    case cudaErrorIllegalAddress:
                    case cudaErrorInvalidPtx:
                    case cudaErrorInvalidGraphicsContext:
                    case cudaErrorStartupFailure:
                        throw runtime_error{cudaGetErrorString(err)};

                    case cudaErrorLaunchOutOfResources:
                    case cudaErrorInvalidConfiguration:
                    case cudaErrorInvalidDevice:
                    case cudaErrorInvalidValue:
                    case cudaErrorInvalidPitchValue:
                    case cudaErrorInvalidSymbol:
                    case cudaErrorInvalidHostPointer:
                    case cudaErrorInvalidDevicePointer:
                    case cudaErrorInvalidTexture:
                    case cudaErrorInvalidTextureBinding:
                    case cudaErrorInvalidChannelDescriptor:
                    case cudaErrorInvalidMemcpyDirection:
                    case cudaErrorInvalidFilterSetting:
                    case cudaErrorInvalidNormSetting:
                    case cudaErrorInvalidResourceHandle:
                    case cudaErrorSetOnActiveProcess:
                    case cudaErrorInvalidSurface:
                    case cudaErrorUnsupportedLimit:
                    case cudaErrorDuplicateVariableName:
                    case cudaErrorDuplicateTextureName:
                    case cudaErrorDuplicateSurfaceName:
                    case cudaErrorInvalidKernelImage:
                    case cudaErrorNoKernelImageForDevice:
                    case cudaErrorPeerAccessAlreadyEnabled:
                    case cudaErrorPeerAccessNotEnabled:
                    case cudaErrorHostMemoryAlreadyRegistered:
                    case cudaErrorHostMemoryNotRegistered:
                    case cudaErrorLaunchFileScopedTex:
                    case cudaErrorLaunchFileScopedSurf:
                        throw invalid_argument{cudaGetErrorString(err)};

                    case cudaErrorMemoryAllocation:
                        throw bad_alloc{};

                    default:
                        throw invalid_argument{"Unknown error type"};
                }
                #pragma GCC diagnostic pop
            }
        }
    }
}



#endif /* DDRF_CUDA_BITS_THROW_ERROR_H_ */
