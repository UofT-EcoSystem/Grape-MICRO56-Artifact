// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 1999-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */



//***************************** Module Header **********************************
//
// This code is linked into the resource manager proper.  It receives the
//    ioctl from the resource manager's customer, unbundles the args and
//    calls the correct resman routines.
//
//******************************************************************************

#include <core/prelude.h>
#include <core/locks.h>
#include <nv.h>
#include <nv_escape.h>
#include <osapi.h>
#include <rmapi/exports.h>
#include <nv-unix-nvos-params-wrappers.h>

#include <nvos.h>
#include <class/cl0000.h> // NV01_ROOT
#include <class/cl0001.h> // NV01_ROOT_NON_PRIV
#include <class/cl0005.h> // NV01_EVENT
#include <class/cl003e.h> // NV01_MEMORY_SYSTEM
#include <class/cl0071.h> // NV01_MEMORY_SYSTEM_OS_DESCRIPTOR

#include <ctrl/ctrl00fd.h>

#define NV_CTL_DEVICE_ONLY(nv)                 \
{                                              \
    if (((nv)->flags & NV_FLAG_CONTROL) == 0)  \
    {                                          \
        rmStatus = NV_ERR_INVALID_ARGUMENT;    \
        goto done;                             \
    }                                          \
}

#define NV_ACTUAL_DEVICE_ONLY(nv)              \
{                                              \
    if (((nv)->flags & NV_FLAG_CONTROL) != 0)  \
    {                                          \
        rmStatus = NV_ERR_INVALID_ARGUMENT;    \
        goto done;                             \
    }                                          \
}

static NvBool RmIsDeviceRefNeeded(NVOS54_PARAMETERS *pApi)
{
    switch(pApi->cmd)
    {
        case NV00FD_CTRL_CMD_ATTACH_MEM:
            return NV_TRUE;
        default:
            return NV_FALSE;
    }
}

static NV_STATUS RmGetDeviceFd(NVOS54_PARAMETERS *pApi, NvS32 *pFd)
{
    RMAPI_PARAM_COPY paramCopy;
    void *pKernelParams;
    NvU32 paramSize;
    NV_STATUS status;

    *pFd = -1;

    switch(pApi->cmd)
    {
        case NV00FD_CTRL_CMD_ATTACH_MEM:
            paramSize = sizeof(NV00FD_CTRL_ATTACH_MEM_PARAMS);
            break;
        default:
            return NV_ERR_INVALID_ARGUMENT;
    }

    RMAPI_PARAM_COPY_INIT(paramCopy, pKernelParams, pApi->params, paramSize, 1);

    status = rmapiParamsAcquire(&paramCopy, NV_TRUE);
    if (status != NV_OK)
        return status;

    switch(pApi->cmd)
    {
        case NV00FD_CTRL_CMD_ATTACH_MEM:
            *pFd = (NvS32)((NV00FD_CTRL_ATTACH_MEM_PARAMS *)pKernelParams)->devDescriptor;
            break;
        default:
            NV_ASSERT(0);
            break;
    }

    NV_ASSERT(rmapiParamsRelease(&paramCopy) == NV_OK);

    return status;
}

// Only return errors through pApi->status
static void RmCreateOsDescriptor(NVOS32_PARAMETERS *pApi, API_SECURITY_INFO secInfo)
{
    NV_STATUS rmStatus;
    NvBool writable;
    NvU32 flags = 0;
    NvU64 allocSize, pageCount, *pPteArray = NULL;
    void *pDescriptor, *pPageArray = NULL;

    pDescriptor = NvP64_VALUE(pApi->data.AllocOsDesc.descriptor);
    if (((NvUPtr)pDescriptor & ~os_page_mask) != 0)
    {
        rmStatus = NV_ERR_NOT_SUPPORTED;
        goto done;
    }

    // Check to prevent an NvU64 overflow
    if ((pApi->data.AllocOsDesc.limit + 1) == 0)
    {
        rmStatus = NV_ERR_INVALID_LIMIT;
        goto done;
    }

    allocSize = (pApi->data.AllocOsDesc.limit + 1);
    pageCount = (1 + ((allocSize - 1) / os_page_size));

    writable = FLD_TEST_DRF(OS32, _ATTR2, _PROTECTION_USER, _READ_WRITE, pApi->data.AllocOsDesc.attr2);

    flags = FLD_SET_DRF_NUM(_LOCK_USER_PAGES, _FLAGS, _WRITE, writable, flags);
    rmStatus = os_lock_user_pages(pDescriptor, pageCount, &pPageArray, flags);
    if (rmStatus == NV_OK)
    {
        pApi->data.AllocOsDesc.descriptor = (NvP64)(NvUPtr)pPageArray;
        pApi->data.AllocOsDesc.descriptorType = NVOS32_DESCRIPTOR_TYPE_OS_PAGE_ARRAY;
    }
    else if (rmStatus == NV_ERR_INVALID_ADDRESS)
    {
        rmStatus = os_lookup_user_io_memory(pDescriptor, pageCount,
                &pPteArray, &pPageArray);
        if (rmStatus == NV_OK)
        {
            if (pPageArray != NULL)
            {
                pApi->data.AllocOsDesc.descriptor = (NvP64)(NvUPtr)pPageArray;
                pApi->data.AllocOsDesc.descriptorType = NVOS32_DESCRIPTOR_TYPE_OS_PAGE_ARRAY;
            }
            else if (pPteArray != NULL)
            {
                pApi->data.AllocOsDesc.descriptor = (NvP64)(NvUPtr)pPteArray;
                pApi->data.AllocOsDesc.descriptorType = NVOS32_DESCRIPTOR_TYPE_OS_IO_MEMORY;
            }
            else
            {
                NV_ASSERT_FAILED("unknown memory import type");
                rmStatus = NV_ERR_NOT_SUPPORTED;
            }
        }
    }
    if (rmStatus != NV_OK)
        goto done;

    Nv04VidHeapControlWithSecInfo(pApi, secInfo);

    if (pApi->status != NV_OK)
    {
        switch (pApi->data.AllocOsDesc.descriptorType)
        {
            default:
                break;
            case NVOS32_DESCRIPTOR_TYPE_OS_PAGE_ARRAY:
                os_unlock_user_pages(pageCount, pPageArray);
                break;
        }
    }

done:
    if (rmStatus != NV_OK)
        pApi->status = rmStatus;
}

// Only return errors through pApi->status
static void RmAllocOsDescriptor(NVOS02_PARAMETERS *pApi, API_SECURITY_INFO secInfo)
{
    NV_STATUS rmStatus = NV_OK;
    NvU32 flags, attr, attr2;
    NVOS32_PARAMETERS *pVidHeapParams;

    if (!FLD_TEST_DRF(OS02, _FLAGS, _LOCATION, _PCI, pApi->flags) ||
        !FLD_TEST_DRF(OS02, _FLAGS, _MAPPING, _NO_MAP, pApi->flags))
    {
        rmStatus = NV_ERR_INVALID_FLAGS;
        goto done;
    }

    attr = DRF_DEF(OS32, _ATTR, _LOCATION, _PCI);

    if (FLD_TEST_DRF(OS02, _FLAGS, _COHERENCY, _CACHED, pApi->flags) ||
        FLD_TEST_DRF(OS02, _FLAGS, _COHERENCY, _WRITE_BACK, pApi->flags))
    {
        attr = FLD_SET_DRF(OS32, _ATTR, _COHERENCY, _WRITE_BACK, attr);
    }
    else if (FLD_TEST_DRF(OS02, _FLAGS, _COHERENCY, _UNCACHED, pApi->flags))
        attr = FLD_SET_DRF(OS32, _ATTR, _COHERENCY, _UNCACHED, attr);
    else {
        rmStatus = NV_ERR_INVALID_FLAGS;
        goto done;
    }

    if (FLD_TEST_DRF(OS02, _FLAGS, _PHYSICALITY, _CONTIGUOUS, pApi->flags))
        attr = FLD_SET_DRF(OS32, _ATTR, _PHYSICALITY, _CONTIGUOUS, attr);
    else
        attr = FLD_SET_DRF(OS32, _ATTR, _PHYSICALITY, _NONCONTIGUOUS, attr);

    if (FLD_TEST_DRF(OS02, _FLAGS, _GPU_CACHEABLE, _YES, pApi->flags))
        attr2 = DRF_DEF(OS32, _ATTR2, _GPU_CACHEABLE, _YES);
    else
        attr2 = DRF_DEF(OS32, _ATTR2, _GPU_CACHEABLE, _NO);

    pVidHeapParams = portMemAllocNonPaged(sizeof(NVOS32_PARAMETERS));
    if (pVidHeapParams == NULL)
    {
        rmStatus = NV_ERR_NO_MEMORY;
        goto done;
    }
    portMemSet(pVidHeapParams, 0, sizeof(NVOS32_PARAMETERS));

    pVidHeapParams->hRoot = pApi->hRoot;
    pVidHeapParams->hObjectParent = pApi->hObjectParent;
    pVidHeapParams->function = NVOS32_FUNCTION_ALLOC_OS_DESCRIPTOR;

    flags = (NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED |
             NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED);

    if (DRF_VAL(OS02, _FLAGS, _ALLOC_USER_READ_ONLY, pApi->flags))
        attr2 = FLD_SET_DRF(OS32, _ATTR2, _PROTECTION_USER, _READ_ONLY, attr2);

    // Currently CPU-RO memory implies GPU-RO as well
    if (DRF_VAL(OS02, _FLAGS, _ALLOC_DEVICE_READ_ONLY, pApi->flags) ||
        DRF_VAL(OS02, _FLAGS, _ALLOC_USER_READ_ONLY, pApi->flags))
        attr2 = FLD_SET_DRF(OS32, _ATTR2, _PROTECTION_DEVICE, _READ_ONLY, attr2);

    pVidHeapParams->data.AllocOsDesc.hMemory = pApi->hObjectNew;
    pVidHeapParams->data.AllocOsDesc.flags = flags;
    pVidHeapParams->data.AllocOsDesc.attr = attr;
    pVidHeapParams->data.AllocOsDesc.attr2 = attr2;
    pVidHeapParams->data.AllocOsDesc.descriptor = pApi->pMemory;
    pVidHeapParams->data.AllocOsDesc.limit = pApi->limit;
    pVidHeapParams->data.AllocOsDesc.descriptorType = NVOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS;

    RmCreateOsDescriptor(pVidHeapParams, secInfo);

    pApi->status = pVidHeapParams->status;

    portMemFree(pVidHeapParams);

done:
    if (rmStatus != NV_OK)
        pApi->status = rmStatus;
}

ct_assert(NV_OFFSETOF(NVOS21_PARAMETERS, hRoot) == NV_OFFSETOF(NVOS64_PARAMETERS, hRoot));
ct_assert(NV_OFFSETOF(NVOS21_PARAMETERS, hObjectParent) == NV_OFFSETOF(NVOS64_PARAMETERS, hObjectParent));
ct_assert(NV_OFFSETOF(NVOS21_PARAMETERS, hObjectNew) == NV_OFFSETOF(NVOS64_PARAMETERS, hObjectNew));
ct_assert(NV_OFFSETOF(NVOS21_PARAMETERS, hClass) == NV_OFFSETOF(NVOS64_PARAMETERS, hClass));
ct_assert(NV_OFFSETOF(NVOS21_PARAMETERS, pAllocParms) == NV_OFFSETOF(NVOS64_PARAMETERS, pAllocParms));

// <bojian/Grape>
// clang-format on

#include "NVCapturePMAAllocMode.h"

static NVCapturePMAAllocMode_t sCapturePMAAllocMode = kDefault;

typedef struct {
	NVOS32_PARAMETERS alloc_data;
	NvU32 shadow_hmemory;
} VidHeapCtrlDataWithShadowMemoryHandle;

MAKE_LIST(CachedVidHeapControlDataList, VidHeapCtrlDataWithShadowMemoryHandle);
MAKE_LIST(CachedVidHeapControlDataResidualList,
	  VidHeapCtrlDataWithShadowMemoryHandle);

static CachedVidHeapControlDataList sCachedVidHeapControlDataList;
static CachedVidHeapControlDataListIter
	sCachedVidHeapControlDataList_it = {},
	sCachedVidHeapControlDataList_query_it = {};
static int sCachedVidHeapControlDataListInitialized = 0;

static CachedVidHeapControlDataResidualList
	sCachedVidHeapControlDataResidualList;
static CachedVidHeapControlDataResidualListIter
	sCachedVidHeapControlDataResidualList_it = {};
static int sCachedVidHeapControlDataResidualListInitialized = 0;
static int sResidualWasAllocated = 0;
static NvU32 sCachedVidHeapControlDataResidualList_idx = 0;

NV_STATUS RmQueryRecordedPMAAllocSize_init(void)
{
	sCachedVidHeapControlDataList_query_it =
		listIterAll(&sCachedVidHeapControlDataList);
	return NV_OK;
}

NV_STATUS RmQueryRecordedPMAAllocSize(NvU64 *pma_alloc_size)
{
	if (!listIterNext(&sCachedVidHeapControlDataList_query_it)) {
		return NV_ERR_GENERIC;
	}
	if (sCachedVidHeapControlDataList_query_it.pValue == NULL) {
		return NV_ERR_GENERIC;
	}
	*pma_alloc_size = sCachedVidHeapControlDataList_query_it.pValue
				  ->alloc_data.data.AllocSize.size;
	return NV_OK;
}

NV_STATUS RmQueryNumRecordedResiduals(NvU32 *residual_capacity,
				      NvU32 *residual_idx)
{
	*residual_capacity = listCount(&sCachedVidHeapControlDataResidualList);
	*residual_idx = sCachedVidHeapControlDataResidualList_idx;
	return NV_OK;
}

NV_STATUS RmQueryCapturePMAAllocMode(NvU32 *capture_pma_alloc_mode)
{
	*capture_pma_alloc_mode = sCapturePMAAllocMode;
	return NV_OK;
}

NV_STATUS RmCapturePMAAlloc(NvU32 capture_pma_alloc_mode)
{
	// This should not happen, as the file write operation already checks
	// for the validity of the mode value, but we still add it just in case.
	if (capture_pma_alloc_mode >= kEnd) {
		return NV_ERR_GENERIC;
	}
	sCapturePMAAllocMode = capture_pma_alloc_mode;

	if (sCapturePMAAllocMode == kReplay ||
	    sCapturePMAAllocMode == kReplayNext ||
	    sCapturePMAAllocMode == kReplayNextAndStashResiduals ||
	    sCapturePMAAllocMode == kReplayNextAndAppendResiduals) {
		if (sCachedVidHeapControlDataList_it.pValue == NULL) {
#define ITERATOR_TO_LIST_BEGIN(iterator, iterator_type, list)             \
	NV_PRINTF(LEVEL_ERROR,                                            \
		  "Pointing the iterator to the head of the list\n");     \
	iterator_type it = listIterAll(&list);                            \
	NV_PRINTF(LEVEL_ERROR,                                            \
		  "Replaying the previously recorded %d allocations [\n", \
		  listCount(&list));                                      \
	while (listIterNext(&it)) {                                       \
		NV_PRINTF(LEVEL_ERROR, "  .size=%lld\n",                  \
			  it.pValue->alloc_data.data.AllocSize.size);     \
	}                                                                 \
	NV_PRINTF(LEVEL_ERROR, "]\n");                                    \
	iterator = listIterAll(&list)

			ITERATOR_TO_LIST_BEGIN(sCachedVidHeapControlDataList_it,
					       CachedVidHeapControlDataListIter,
					       sCachedVidHeapControlDataList);
			if (!listIterNext(&sCachedVidHeapControlDataList_it)) {
				NV_PRINTF(
					LEVEL_ERROR,
					"Cached list is empty. Directly exiting.\n");
				sCapturePMAAllocMode = kDefault;
				return NV_ERR_GENERIC;
			}
		} else {
			CachedVidHeapControlDataListIter it =
				sCachedVidHeapControlDataList_it;
			NV_PRINTF(LEVEL_ERROR,
				  "Continuing on the current iterator [\n");
			NV_PRINTF(LEVEL_ERROR, "  ...\n");
			do {
				NV_PRINTF(LEVEL_ERROR, "  .size=%lld\n",
					  it.pValue->alloc_data.data.AllocSize
						  .size);
			} while (listIterNext(&it));
			NV_PRINTF(LEVEL_ERROR, "]\n");
		}
	} else if (sCapturePMAAllocMode == kReplayResiduals) {
		ITERATOR_TO_LIST_BEGIN(sCachedVidHeapControlDataResidualList_it,
				       CachedVidHeapControlDataResidualListIter,
				       sCachedVidHeapControlDataResidualList);
		if (!listIterNext(&sCachedVidHeapControlDataResidualList_it)) {
			sCachedVidHeapControlDataResidualList_it.pValue = NULL;
		}

		sCachedVidHeapControlDataResidualList_idx = 0;

	} else if (sCapturePMAAllocMode == kClearRecords) {
#define DESTRUCT_LIST(list)                                                   \
	if (list##Initialized) {                                              \
		NV_PRINTF(LEVEL_ERROR, "Deconstructing the cached list\n");   \
		listDestroy(&list);                                           \
		list##_it.pValue = NULL;                                      \
		list##Initialized = 0;                                        \
		sCapturePMAAllocMode = kDefault;                              \
	} else {                                                              \
		NV_PRINTF(LEVEL_ERROR, "The cached list is not initialized, " \
				       "hence directly exiting\n");           \
	}

		DESTRUCT_LIST(sCachedVidHeapControlDataList);

	} else if (sCapturePMAAllocMode == kClearListOfResiduals) {
		DESTRUCT_LIST(sCachedVidHeapControlDataResidualList);
		sCachedVidHeapControlDataResidualList_idx = 0;
	}
	return NV_OK;
}

NV_STATUS RmDupMemoryCallback(void)
{
	if (sCapturePMAAllocMode == kReplay) {
		if (!listIterNext(&sCachedVidHeapControlDataList_it)) {
			NV_PRINTF(LEVEL_ERROR,
				  "Running out of the recorded allocations\n");
			sCachedVidHeapControlDataList_it.pValue = NULL;
			sCapturePMAAllocMode = kDefault;
		}
	} else if (sCapturePMAAllocMode == kReplayNext) {
		if (!listIterNext(&sCachedVidHeapControlDataList_it)) {
			sCachedVidHeapControlDataList_it.pValue = NULL;
		}
		sCapturePMAAllocMode = kDefault;
	} else if (sCapturePMAAllocMode == kReplayNextAndStashResiduals) {
		if (!listIterNext(&sCachedVidHeapControlDataList_it)) {
			sCachedVidHeapControlDataList_it.pValue = NULL;
		}
		sCapturePMAAllocMode = kReplayResiduals;

		ITERATOR_TO_LIST_BEGIN(sCachedVidHeapControlDataResidualList_it,
				       CachedVidHeapControlDataResidualListIter,
				       sCachedVidHeapControlDataResidualList);
		if (!listIterNext(&sCachedVidHeapControlDataResidualList_it)) {
			sCachedVidHeapControlDataResidualList_it.pValue = NULL;
		}

		sCachedVidHeapControlDataResidualList_idx = 0;

	} else if (sCapturePMAAllocMode == kReplayNextAndAppendResiduals) {
		if (!listIterNext(&sCachedVidHeapControlDataList_it)) {
			sCachedVidHeapControlDataList_it.pValue = NULL;
		}
		sCapturePMAAllocMode = kReplayResiduals;

		// Do NOT reset the iterator of the residual list.
	} else if (sCapturePMAAllocMode == kReplayResiduals &&
		   sResidualWasAllocated) {
		sResidualWasAllocated = 0;
		if (!listIterNext(&sCachedVidHeapControlDataResidualList_it)) {
			NV_PRINTF(LEVEL_ERROR,
				  "Running out of the recorded allocations\n");
			sCachedVidHeapControlDataResidualList_it.pValue = NULL;
			// Do not go to default mode yet, since the number of
			// residuals could grow dynamically.
			//
			//     sCapturePMAAllocMode = kDefault;
		}

		NV_PRINTF(
			LEVEL_ERROR,
			"sCachedVidHeapControlDataResidualList_idx: %d -> %d\n",
			sCachedVidHeapControlDataResidualList_idx,
			sCachedVidHeapControlDataResidualList_idx + 1);
		sCachedVidHeapControlDataResidualList_idx += 1;

		// sCachedVidHeapControlDataResidualList_it =
		// 	sCachedVidHeapControlDataResidualList_it_next;
	}
	return NV_OK;
}

NV_STATUS RmMapToMaterializedhMemory(NvHandle shadow_hmemory,
				     NvHandle *materialized_hmemory)
{
	if (sCapturePMAAllocMode == kReplay ||
	    sCapturePMAAllocMode == kReplayNext ||
	    sCapturePMAAllocMode == kReplayNextAndStashResiduals ||
	    sCapturePMAAllocMode == kReplayNextAndAppendResiduals) {
#define GET_MATERIALIZED_HMEMORY_FROM_ITERATOR(iterator)                    \
	if (iterator.pValue != NULL &&                                      \
	    shadow_hmemory == iterator.pValue->shadow_hmemory) {            \
		*materialized_hmemory =                                     \
			iterator.pValue->alloc_data.data.AllocSize.hMemory; \
		return NV_OK;                                               \
	} else {                                                            \
		NV_PRINTF(LEVEL_ERROR,                                      \
			  "Shadow memory handle=0x%x not found\n",          \
			  shadow_hmemory);                                  \
	}

		GET_MATERIALIZED_HMEMORY_FROM_ITERATOR(
			sCachedVidHeapControlDataList_it);

	} // if (sCapturePMAAllocMode == kReplay ||
	  //     sCapturePMAAllocMode == kReplayNext ||
	  //     sCapturePMAAllocMode == kReplayNextAndStashResiduals ||
	  //     sCapturePMAAllocMode == kReplayNextAndAppendResiduals)
	else if (sCapturePMAAllocMode == kReplayResiduals) {
		GET_MATERIALIZED_HMEMORY_FROM_ITERATOR(
			sCachedVidHeapControlDataResidualList_it);
	}
	return NV_ERR_GENERIC;
}

// clang-format off
// </bojian/Grape>

NV_STATUS RmIoctl(
    nv_state_t  *nv,
    nv_file_private_t *nvfp,
    NvU32        cmd,
    void        *data,
    NvU32        dataSize
)
{
    NV_STATUS            rmStatus = NV_ERR_GENERIC;
    API_SECURITY_INFO    secInfo = { };

    // <bojian/Grape>
	// clang-format on

	// Check whether the memory allocations belong to residual, each having
	// the size of 1 MB.
#define NV_RESIDUAL_MALLOC_SIZE (1 * 1024 * 1024)
#define NV_HUGE_PAGE_SIZE (2 * 1024 * 1024)

	VidHeapCtrlDataWithShadowMemoryHandle cached_vid_heap_control_data = {};

	// clang-format off
    // </bojian/Grape>

    secInfo.privLevel = osIsAdministrator() ? RS_PRIV_LEVEL_USER_ROOT : RS_PRIV_LEVEL_USER;
    secInfo.paramLocation = PARAM_LOCATION_USER;
    secInfo.pProcessToken = NULL;
    secInfo.gpuOsInfo = NULL;
    secInfo.clientOSInfo = nvfp->ctl_nvfp;
    if (secInfo.clientOSInfo == NULL)
        secInfo.clientOSInfo = nvfp;

    switch (cmd)
    {
        case NV_ESC_RM_ALLOC_MEMORY:
        {
            nv_ioctl_nvos02_parameters_with_fd *pApi;
            NVOS02_PARAMETERS *pParms;

            pApi = data;
            pParms = &pApi->params;

            NV_ACTUAL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(nv_ioctl_nvos02_parameters_with_fd))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            if (pParms->hClass == NV01_MEMORY_SYSTEM_OS_DESCRIPTOR)
                RmAllocOsDescriptor(pParms, secInfo);
            else
            {
                NvU32 flags = pParms->flags;

                Nv01AllocMemoryWithSecInfo(pParms, secInfo);

                //
                // If the system memory is going to be mapped immediately,
                // create the mmap context for it now.
                //
                if ((pParms->hClass == NV01_MEMORY_SYSTEM) &&
                    (!FLD_TEST_DRF(OS02, _FLAGS, _ALLOC, _NONE, flags)) &&
                    (!FLD_TEST_DRF(OS02, _FLAGS, _MAPPING, _NO_MAP, flags)) &&
                    (pParms->status == NV_OK))
                {
                    if (rm_create_mmap_context(pParms->hRoot,
                            pParms->hObjectParent, pParms->hObjectNew,
                            pParms->pMemory, pParms->limit + 1, 0,
                            NV_MEMORY_DEFAULT,
                            pApi->fd) != NV_OK)
                    {
                        NV_PRINTF(LEVEL_WARNING,
                                  "could not create mmap context for %p\n",
                                  NvP64_VALUE(pParms->pMemory));
                        rmStatus = NV_ERR_INVALID_ARGUMENT;
                        goto done;
                    }
                }
            }

            break;
        }

        case NV_ESC_RM_ALLOC_OBJECT:
        {
            NVOS05_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS05_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv01AllocObjectWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_RM_ALLOC:
        {
            NVOS21_PARAMETERS *pApi = data;
            NVOS64_PARAMETERS *pApiAccess = data;
            NvBool bAccessApi = (dataSize == sizeof(NVOS64_PARAMETERS));

            if ((dataSize != sizeof(NVOS21_PARAMETERS)) &&
                (dataSize != sizeof(NVOS64_PARAMETERS)))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            switch (pApi->hClass)
            {
                case NV01_ROOT:
                case NV01_ROOT_CLIENT:
                case NV01_ROOT_NON_PRIV:
                {
                    NV_CTL_DEVICE_ONLY(nv);

                    // Force userspace client allocations to be the _CLIENT class.
                    pApi->hClass = NV01_ROOT_CLIENT;
                    break;
                }
                case NV01_EVENT:
                case NV01_EVENT_OS_EVENT:
                case NV01_EVENT_KERNEL_CALLBACK:
                case NV01_EVENT_KERNEL_CALLBACK_EX:
                {
                    break;
                }
                default:
                {
                    NV_CTL_DEVICE_ONLY(nv);
                    break;
                }
            }

            if (!bAccessApi)
            {
                Nv04AllocWithSecInfo(pApi, secInfo);
            }
            else
            {
                Nv04AllocWithAccessSecInfo(pApiAccess, secInfo);
            }

            break;
        }

        case NV_ESC_RM_FREE:
        {
            NVOS00_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS00_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

        // <bojian/Grape>
		// clang-format on

		NVOS00_PARAMETERS old_param = *pApi;

		// clang-format off
        // </bojian/Grape>

            Nv01FreeWithSecInfo(pApi, secInfo);

        // <bojian/Grape>
		// clang-format on

#define TRACK_pAPI_CHANGES(new_member, old_member, fmt_str)          \
	if (new_member != old_member) {                              \
		NV_PRINTF(LEVEL_ERROR,                               \
			  #old_member "=" fmt_str " -> " #new_member \
				      "=" fmt_str "\n",              \
			  old_member, new_member);                   \
	}

		// Dump the changes after the `Nv01FreeWithSecInfo` API call.
		TRACK_pAPI_CHANGES(pApi->hObjectOld, old_param.hObjectOld,
				   "%d");
		TRACK_pAPI_CHANGES(pApi->hObjectParent, old_param.hObjectParent,
				   "%d");
		TRACK_pAPI_CHANGES(pApi->hRoot, old_param.hRoot, "%d");
		TRACK_pAPI_CHANGES(pApi->status, old_param.status, "%d");

		// clang-format off
        // </bojian/Grape>

            if (pApi->status == NV_OK &&
                pApi->hObjectOld == pApi->hRoot)
            {
                rm_client_free_os_events(pApi->hRoot);
            }

            break;
        }

        case NV_ESC_RM_VID_HEAP_CONTROL:
        {
            NVOS32_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS32_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

        // <bojian/Grape>
		// clang-format on

		NVOS32_PARAMETERS old_params = *pApi;

// Check whether the attribute matches the given value.
#define CHECK_ATTR_NE_VALUE(attr, bits_begin, bits_end, value) \
	((((attr) << (31 - (bits_begin))) >>                   \
	  (31 + (bits_end) - (bits_begin))) ^                  \
	 value)

#define CHECK_ATTR_EQ_VALUE(attr, bits_begin, bits_end, value) \
	(!CHECK_ATTR_NE_VALUE(attr, bits_begin, bits_end, value))

		/// @sa src/common/sdk/nvidia/inc/nvos.h
		NvU32 pApiUsesDefaultPageSize =
			CHECK_ATTR_EQ_VALUE(pApi->data.AllocSize.attr, 24, 23,
					    NVOS32_ATTR_PAGE_SIZE_DEFAULT);
		NvU32 pApiIsLocatedOnVidMem =
			CHECK_ATTR_EQ_VALUE(pApi->data.AllocSize.attr, 26, 25,
					    NVOS32_ATTR_LOCATION_VIDMEM);

		if ((sCapturePMAAllocMode == kReplay ||
		     sCapturePMAAllocMode == kReplayNext ||
		     sCapturePMAAllocMode == kReplayNextAndStashResiduals ||
		     sCapturePMAAllocMode == kReplayNextAndAppendResiduals) &&
		    pApiUsesDefaultPageSize && pApiIsLocatedOnVidMem) {
			NV_PRINTF(LEVEL_ERROR,
				  "Taping out the recorded parameters\n");
			if (sCachedVidHeapControlDataList_it.pValue == NULL) {
				NV_PRINTF(
					LEVEL_ERROR,
					"Running out of cached allocations. Directly exiting.\n");
				break;
			}

#define COPY_ITERATOR_TO_pAPI(iterator)                                     \
	pApi->data.AllocSize.attr =                                         \
		iterator.pValue->alloc_data.data.AllocSize.attr;            \
	pApi->data.AllocSize.format =                                       \
		iterator.pValue->alloc_data.data.AllocSize.format;          \
	pApi->data.AllocSize.partitionStride =                              \
		iterator.pValue->alloc_data.data.AllocSize.partitionStride; \
	pApi->data.AllocSize.offset =                                       \
		iterator.pValue->alloc_data.data.AllocSize.offset;          \
	pApi->data.AllocSize.limit =                                        \
		iterator.pValue->alloc_data.data.AllocSize.limit;           \
	pApi->data.AllocSize.attr2 =                                        \
		iterator.pValue->alloc_data.data.AllocSize.attr2;           \
	iterator.pValue->shadow_hmemory = pApi->data.AllocSize.hMemory;     \
	break

			COPY_ITERATOR_TO_pAPI(sCachedVidHeapControlDataList_it);
		}

		if (sCapturePMAAllocMode == kReplayResiduals &&
		    pApiUsesDefaultPageSize && pApiIsLocatedOnVidMem) {
			if (sCachedVidHeapControlDataResidualList_it.pValue !=
			    NULL) {
				sResidualWasAllocated = 1;
				COPY_ITERATOR_TO_pAPI(
					sCachedVidHeapControlDataResidualList_it);
			}
		}

		if (sCapturePMAAllocMode == kProbeNext &&
		    pApiUsesDefaultPageSize && pApiIsLocatedOnVidMem) {
			NV_PRINTF(LEVEL_ERROR,
				  "Probing the size of the allocation\n");
			cached_vid_heap_control_data.alloc_data = *pApi;

			if (!sCachedVidHeapControlDataListInitialized) {
				listInit(&sCachedVidHeapControlDataList,
					 portMemAllocatorGetGlobalNonPaged());
				sCachedVidHeapControlDataListInitialized = 1;
			}
			if (listAppendValue(&sCachedVidHeapControlDataList,
					    &cached_vid_heap_control_data) ==
			    NULL) {
				NV_PRINTF(
					LEVEL_ERROR,
					"Failed to insert to the list due to insufficient resources\n");
			}
			sCapturePMAAllocMode = kDefault;
			break;
		}

#define NV_ALIGN_RESIDUALS_TO_HUGE_PAGE 0

		// Change the size of residual allocations from 1 MB to align
		// with the huge page size (i.e., 2 MB) so that they could be
		// replayed using the `cudaMalloc` calls.
#if NV_ALIGN_RESIDUALS_TO_HUGE_PAGE
		if (sCapturePMAAllocMode == kRecord &&
		    pApiUsesDefaultPageSize && pApiIsLocatedOnVidMem &&
		    listCount(&sCachedVidHeapControlDataList) >= 1) {
			if (pApi->data.AllocSize.size ==
			    NV_RESIDUAL_MALLOC_SIZE) {
				pApi->data.AllocSize.size = NV_HUGE_PAGE_SIZE;
			}
		}
#endif // defined(NV_ALIGN_RESIDUALS_TO_HUGE_PAGE)

		// clang-format off
        // </bojian/Grape>

            if (pApi->function == NVOS32_FUNCTION_ALLOC_OS_DESCRIPTOR)
                RmCreateOsDescriptor(pApi, secInfo);
            else
                Nv04VidHeapControlWithSecInfo(pApi, secInfo);

        // <bojian/Grape>
		// clang-format on

		if ((sCapturePMAAllocMode == kRecord ||
		     sCapturePMAAllocMode == kRecordNextAndOverwrite ||
		     sCapturePMAAllocMode == kReplayResiduals) &&
		    pApiUsesDefaultPageSize && pApiIsLocatedOnVidMem) {
			// Here we only track changes of members that have been
			// marked as [OUT],
			TRACK_pAPI_CHANGES(pApi->status, old_params.status,
					   "%d");
			TRACK_pAPI_CHANGES(pApi->total, old_params.total,
					   "%lld");
			TRACK_pAPI_CHANGES(pApi->free, old_params.free, "%lld");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.hMemory,
					   old_params.data.AllocSize.hMemory,
					   "0x%x");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.attr,
					   old_params.data.AllocSize.attr,
					   "0x%x");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.format,
					   old_params.data.AllocSize.format,
					   "0x%x");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.comprCovg,
					   old_params.data.AllocSize.comprCovg,
					   "%d");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.zcullCovg,
					   old_params.data.AllocSize.zcullCovg,
					   "%d");
			TRACK_pAPI_CHANGES(
				pApi->data.AllocSize.partitionStride,
				old_params.data.AllocSize.partitionStride,
				"%d");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.size,
					   old_params.data.AllocSize.size,
					   "%lld");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.offset,
					   old_params.data.AllocSize.offset,
					   "0x%llx");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.limit,
					   old_params.data.AllocSize.limit,
					   "%lld");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.address,
					   old_params.data.AllocSize.address,
					   "%p");
			TRACK_pAPI_CHANGES(pApi->data.AllocSize.attr2,
					   old_params.data.AllocSize.attr2,
					   "0x%x");
			cached_vid_heap_control_data.alloc_data = *pApi;

			if (sCapturePMAAllocMode == kRecord) {
#define INITIALIZE_LIST_AND_APPEND_VALUE(list)                                           \
	if (!list##Initialized) {                                                        \
		listInit(&list, portMemAllocatorGetGlobalNonPaged());                    \
		list##Initialized = 1;                                                   \
	}                                                                                \
	if (listAppendValue(&list, &cached_vid_heap_control_data) == NULL) {             \
		NV_PRINTF(                                                               \
			LEVEL_ERROR,                                                     \
			"Failed to insert to the list due to insufficient resources\n"); \
	}

				INITIALIZE_LIST_AND_APPEND_VALUE(
					sCachedVidHeapControlDataList);
			} else if (sCapturePMAAllocMode ==
				   kRecordNextAndOverwrite) {
				NV_PRINTF(
					LEVEL_ERROR,
					"Attempting to modify the previous record\n");

				if (sCachedVidHeapControlDataList_it.pValue ==
				    NULL) {
					NV_PRINTF(
						LEVEL_ERROR,
						"Pointing the iterator to the head of the list\n");
					ITERATOR_TO_LIST_BEGIN(
						sCachedVidHeapControlDataList_it,
						CachedVidHeapControlDataListIter,
						sCachedVidHeapControlDataList);
					if (!listIterNext(
						    &sCachedVidHeapControlDataList_it)) {
						NV_PRINTF(
							LEVEL_ERROR,
							"Cached list is empty. Directly exiting.\n");
						sCapturePMAAllocMode = kDefault;
						return NV_ERR_GENERIC;
					}
				} // sCachedVidHeapControlDataList_it.pValue ==
				  // NULL
				sCachedVidHeapControlDataList_it.pValue
					->alloc_data = *pApi;
				sCapturePMAAllocMode = kDefault;
			} else if (sCapturePMAAllocMode == kReplayResiduals &&
				   pApi->data.AllocSize.size ==
					   2 * 1024 * 1024) {
				NV_PRINTF(
					LEVEL_ERROR,
					"Appending to the current list of residuals\n");
				sResidualWasAllocated = 1;

				INITIALIZE_LIST_AND_APPEND_VALUE(
					sCachedVidHeapControlDataResidualList);

				sCachedVidHeapControlDataResidualList_it = listIterAll(
					&sCachedVidHeapControlDataResidualList);
				CachedVidHeapControlDataResidualListIter
					sCachedVidHeapControlDataResidualList_it_next =
						sCachedVidHeapControlDataResidualList_it;
				while (listIterNext(
					&sCachedVidHeapControlDataResidualList_it_next)) {
					sCachedVidHeapControlDataResidualList_it =
						sCachedVidHeapControlDataResidualList_it_next;
				}
			} // if (sCapturePMAAllocMode == kRecord)
		} // if ((sCapturePMAAllocMode == kRecord ||
		  //      sCapturePMAAllocMode == kReplayResiduals) &&
		  //     pApiUsesDefaultPageSize && pApiIsLocatedOnVidMem)

#undef TRACK_pAPI_CHANGES

		// clang-format off
        // </bojian/Grape>

            break;
        }

        case NV_ESC_RM_I2C_ACCESS:
        {
            NVOS_I2C_ACCESS_PARAMS *pApi = data;

            NV_ACTUAL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS_I2C_ACCESS_PARAMS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv04I2CAccessWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_RM_IDLE_CHANNELS:
        {
            NVOS30_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS30_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv04IdleChannelsWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_RM_MAP_MEMORY:
        {
            nv_ioctl_nvos33_parameters_with_fd *pApi;
            NVOS33_PARAMETERS *pParms;

            pApi = data;
            pParms = &pApi->params;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(nv_ioctl_nvos33_parameters_with_fd))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            // Don't allow userspace to override the caching type
            pParms->flags = FLD_SET_DRF(OS33, _FLAGS, _CACHING_TYPE, _DEFAULT, pParms->flags);
            Nv04MapMemoryWithSecInfo(pParms, secInfo);

            if (pParms->status == NV_OK)
            {
                pParms->status = rm_create_mmap_context(pParms->hClient,
                                 pParms->hDevice, pParms->hMemory,
                                 pParms->pLinearAddress, pParms->length,
                                 pParms->offset,
                                 DRF_VAL(OS33, _FLAGS, _CACHING_TYPE, pParms->flags),
                                 pApi->fd);
                if (pParms->status != NV_OK)
                {
                    NVOS34_PARAMETERS params;
                    portMemSet(&params, 0, sizeof(NVOS34_PARAMETERS));
                    params.hClient        = pParms->hClient;
                    params.hDevice        = pParms->hDevice;
                    params.hMemory        = pParms->hMemory;
                    params.pLinearAddress = pParms->pLinearAddress;
                    params.flags          = pParms->flags;
                    Nv04UnmapMemoryWithSecInfo(&params, secInfo);
                }
            }
            break;
        }

        case NV_ESC_RM_UNMAP_MEMORY:
        {
            NVOS34_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS34_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv04UnmapMemoryWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_RM_ACCESS_REGISTRY:
        {
            NVOS38_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS38_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            pApi->status = rm_access_registry(pApi->hClient,
                                              pApi->hObject,
                                              pApi->AccessType,
                                              pApi->pDevNode,
                                              pApi->DevNodeLength,
                                              pApi->pParmStr,
                                              pApi->ParmStrLength,
                                              pApi->pBinaryData,
                                              &pApi->BinaryDataLength,
                                              &pApi->Data,
                                              &pApi->Entry);
            break;
        }

        case NV_ESC_RM_ALLOC_CONTEXT_DMA2:
        {
            NVOS39_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS39_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv04AllocContextDmaWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_RM_BIND_CONTEXT_DMA:
        {
            NVOS49_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS49_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv04BindContextDmaWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_RM_MAP_MEMORY_DMA:
        {
            NVOS46_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS46_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv04MapMemoryDmaWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_RM_UNMAP_MEMORY_DMA:
        {
            NVOS47_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS47_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv04UnmapMemoryDmaWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_RM_DUP_OBJECT:
        {
            NVOS55_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS55_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv04DupObjectWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_RM_SHARE:
        {
            NVOS57_PARAMETERS *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS57_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            Nv04ShareWithSecInfo(pApi, secInfo);
            break;
        }

        case NV_ESC_ALLOC_OS_EVENT:
        {
            nv_ioctl_alloc_os_event_t *pApi = data;

            if (dataSize != sizeof(nv_ioctl_alloc_os_event_t))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            pApi->Status = rm_alloc_os_event(pApi->hClient,
                                             nvfp,
                                             pApi->fd);
            break;
        }

        case NV_ESC_FREE_OS_EVENT:
        {
            nv_ioctl_free_os_event_t *pApi = data;

            if (dataSize != sizeof(nv_ioctl_free_os_event_t))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            pApi->Status = rm_free_os_event(pApi->hClient, pApi->fd);
            break;
        }

        case NV_ESC_RM_GET_EVENT_DATA:
        {
            NVOS41_PARAMETERS *pApi = data;

            if (dataSize != sizeof(NVOS41_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            pApi->status = rm_get_event_data(nvfp,
                                             pApi->pEvent,
                                             &pApi->MoreEvents);
            break;
        }

        case NV_ESC_STATUS_CODE:
        {
            nv_state_t *pNv;
            nv_ioctl_status_code_t *pApi = data;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(nv_ioctl_status_code_t))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            pNv = nv_get_adapter_state(pApi->domain, pApi->bus, pApi->slot);
            if (pNv == NULL)
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            rmStatus = rm_get_adapter_status(pNv, &pApi->status);

            if (rmStatus != NV_OK)
                goto done;

            break;
        }

        case NV_ESC_RM_CONTROL:
        {
            NVOS54_PARAMETERS *pApi = data;
            void *priv = NULL;
            nv_file_private_t *dev_nvfp = NULL;
            NvS32 fd;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS54_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            if (RmIsDeviceRefNeeded(pApi))
            {
                rmStatus = RmGetDeviceFd(pApi, &fd);
                if (rmStatus != NV_OK)
                {
                    goto done;
                }

                dev_nvfp = nv_get_file_private(fd, NV_FALSE, &priv);
                if (dev_nvfp == NULL)
                {
                    rmStatus = NV_ERR_INVALID_DEVICE;
                    goto done;
                }

                // Check to avoid cyclic dependency with NV_ESC_REGISTER_FD
                if (!portAtomicCompareAndSwapU32(&dev_nvfp->register_or_refcount,
                                                 NVFP_TYPE_REFCOUNTED,
                                                 NVFP_TYPE_NONE))
                {
                    // Is this already refcounted...
                    if (dev_nvfp->register_or_refcount != NVFP_TYPE_REFCOUNTED)
                    {
                        nv_put_file_private(priv);
                        rmStatus = NV_ERR_IN_USE;
                        goto done;
                    }
                }

                secInfo.gpuOsInfo = priv;
            }

            Nv04ControlWithSecInfo(pApi, secInfo);

            if ((pApi->status != NV_OK) && (priv != NULL))
            {
                //
                // No need to reset `register_or_refcount` as it might be set
                // for previous successful calls. We let it clear with FD close.
                //
                nv_put_file_private(priv);

                secInfo.gpuOsInfo = NULL;
            }

            break;
        }

        case NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO:
        {
            NVOS56_PARAMETERS *pApi = data;
            void *pOldCpuAddress;
            void *pNewCpuAddress;

            NV_CTL_DEVICE_ONLY(nv);

            if (dataSize != sizeof(NVOS56_PARAMETERS))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            pOldCpuAddress = NvP64_VALUE(pApi->pOldCpuAddress);
            pNewCpuAddress = NvP64_VALUE(pApi->pNewCpuAddress);

            pApi->status = rm_update_device_mapping_info(pApi->hClient,
                                                         pApi->hDevice,
                                                         pApi->hMemory,
                                                         pOldCpuAddress,
                                                         pNewCpuAddress);
            break;
        }

        case NV_ESC_REGISTER_FD:
        {
            nv_ioctl_register_fd_t *params = data;
            void *priv = NULL;
            nv_file_private_t *ctl_nvfp;

            if (dataSize != sizeof(nv_ioctl_register_fd_t))
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            // LOCK: acquire API lock
            rmStatus = rmapiLockAcquire(API_LOCK_FLAGS_NONE, RM_LOCK_MODULES_OSAPI);
            if (rmStatus != NV_OK)
                goto done;

            // If there is already a ctl fd registered on this nvfp, fail.
            if (nvfp->ctl_nvfp != NULL)
            {
                // UNLOCK: release API lock
                rmapiLockRelease();
                rmStatus = NV_ERR_INVALID_STATE;
                goto done;
            }

            //
            // Note that this call is valid for both "actual" devices and ctrl
            // devices.  In particular, NV_ESC_ALLOC_OS_EVENT can be used with
            // both types of devices.
            // But, the ctl_fd passed in should always correspond to a control FD.
            //
            ctl_nvfp = nv_get_file_private(params->ctl_fd,
                                           NV_TRUE, /* require ctl fd */
                                           &priv);
            if (ctl_nvfp == NULL)
            {
                // UNLOCK: release API lock
                rmapiLockRelease();
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            // Disallow self-referential links, and disallow links to FDs that
            // themselves have a link.
            if ((ctl_nvfp == nvfp) || (ctl_nvfp->ctl_nvfp != NULL))
            {
                nv_put_file_private(priv);
                // UNLOCK: release API lock
                rmapiLockRelease();
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            // Check to avoid cyclic dependency with device refcounting
            if (!portAtomicCompareAndSwapU32(&nvfp->register_or_refcount,
                                             NVFP_TYPE_REGISTERED,
                                             NVFP_TYPE_NONE))
            {
                nv_put_file_private(priv);
                // UNLOCK: release API lock
                rmapiLockRelease();
                rmStatus = NV_ERR_IN_USE;
                goto done;
            }

            //
            // nvfp->ctl_nvfp is read outside the lock, so set it atomically.
            // Note that once set, this can never be removed until the fd
            // associated with nvfp is closed.  We hold on to 'priv' until the
            // fd is closed, too, to ensure that the fd associated with
            // ctl_nvfp remains valid.
            //
            portAtomicSetSize(&nvfp->ctl_nvfp, ctl_nvfp);
            nvfp->ctl_nvfp_priv = priv;

            // UNLOCK: release API lock
            rmapiLockRelease();

            // NOTE: nv_put_file_private(priv) is not called here.  It MUST be
            // called during cleanup of this nvfp.
            rmStatus = NV_OK;
            break;
        }

        default:
        {
            NV_PRINTF(LEVEL_ERROR, "unknown NVRM ioctl command: 0x%x\n", cmd);
            goto done;
        }
    }

    rmStatus = NV_OK;
done:

    return rmStatus;
}
