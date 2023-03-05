#ifndef NV_CAPTURE_PMA_ALLOC_MODE_H_
#define NV_CAPTURE_PMA_ALLOC_MODE_H_

typedef enum {
	kDefault = 0,
	/// Record all the subsequent allocations.
	kRecord = 1,
	/// Replay the recorded entries on all the subsequence memory allocation
	/// requests.
	kReplay = 2,
	/// Replay the first recorded entry on the next immediate memory
	/// allocation request and return back to the default state.
	kReplayNext = 3,
	/// Probe the allocation size of the next request and directly exit.
	kProbeNext = 4,
	/// Clear the recorded entries.
	kClearRecords = 5,
	/// Replay the first recorded entry on the next immediate memory
	/// allocation request. After that, stash the residuals.
	kReplayNextAndStashResiduals = 6,
	/// Replay the subsequent allocations on the list of residuals.
	kReplayResiduals = 7,
	/// Clear the list of residuals.
	kClearListOfResiduals = 8,
	/// Record the next allocation and overwrite the original iterator.
	kRecordNextAndOverwrite = 9,
	/// Replay the first recorded entry on the next immediate memory
	/// allocation request. After that, append the residuals (i.e., it grows
	/// on the current list of residuals).
	kReplayNextAndAppendResiduals = 10,
	kEnd = 11
} NVCapturePMAAllocMode_t;

static const char *NVCapturePMAAllocMode2CStr[kEnd + 1] = {
	"Default",
	"Record",
	"Replay",
	"ReplayNext",
	"ProbeNext",
	"ClearRecords",
	"ReplayNextAndStashResiduals",
	"ReplayResiduals",
	"ClearListOfResiduals",
	"RecordNextAndOverwrite",
	"ReplayNextAndAppendResiduals",
	"End"
};

#endif // NV_CAPTURE_PMA_ALLOC_MODE_H_
