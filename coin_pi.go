package main

import (
	"fmt"
	"math/bits"
	"math/rand/v2"
	"runtime"
	"sync"
	"time"
)

const (
	TotalSeqCnt = 100_000_000
	MaxFlips    = 1_000_000
)

var ratioLUT [MaxFlips/2 + 1]float64

func init() {
	for t := range ratioLUT {
		ratioLUT[t] = float64(t+1) / float64(2*t+1)
	}
}

type result struct {
	runningRatio float64
	completed    int
}

func worker(sequences int, src *rand.Rand) result {
	var r result
	seq := 0
	bits64 := uint64(0)
	avail := 0

	for seq < sequences {
		if avail == 0 {
			bits64 = src.Uint64()
			avail = 64
		}

		// Fast path: count trailing 1-bits (each = heads on first flip = done).
		ones := bits.TrailingZeros64(^bits64)
		if ones > avail {
			ones = avail
		}
		if need := sequences - seq; ones > need {
			ones = need
		}
		if ones > 0 {
			r.runningRatio += float64(ones)
			r.completed += ones
			seq += ones
			bits64 >>= ones
			avail -= ones
			continue
		}

		// Next bit is 0 (tails). Consume it and handle slow path inline.
		bits64 >>= 1
		avail--
		heads := 0
		total := 1

		// Slow path — process in 8-bit chunks where possible to reduce branch cost
		for total < MaxFlips {
			if avail == 0 {
				bits64 = src.Uint64()
				avail = 64
			}

			// If we need many more flips and have ≥8 bits, process a byte at once
			// using popcount to count heads in the chunk.
			deficit := (total - heads) - heads // tails - heads = how far behind heads is
			remaining := MaxFlips - total
			if avail >= 8 && remaining >= 8 && deficit > 8 {
				// Heads can't possibly overtake tails in the next 8 flips,
				// so just count them in bulk.
				chunk := bits64 & 0xFF
				h := bits.OnesCount64(chunk)
				heads += h
				total += 8
				bits64 >>= 8
				avail -= 8
				continue
			}

			// Bit-by-bit when we're close to crossing
			h := int(bits64 & 1)
			bits64 >>= 1
			avail--
			heads += h
			total++
			tails := total - heads
			if heads > tails {
				r.runningRatio += ratioLUT[tails]
				r.completed++
				break
			}
		}
		seq++
	}
	return r
}

func main() {
	numWorkers := runtime.NumCPU()
	seqPerWorker := TotalSeqCnt / numWorkers
	remainder := TotalSeqCnt % numWorkers

	results := make([]result, numWorkers)
	var wg sync.WaitGroup

	now := time.Now()
	for i := 0; i < numWorkers; i++ {
		n := seqPerWorker
		if i < remainder {
			n++
		}
		wg.Add(1)
		go func(idx, count int) {
			defer wg.Done()
			src := rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
			results[idx] = worker(count, src)
		}(i, n)
	}
	wg.Wait()

	totalCompleted := 0
	totalRatio := 0.0
	for _, r := range results {
		totalCompleted += r.completed
		totalRatio += r.runningRatio
	}

	fmt.Println("time taken:", time.Since(now))
	fmt.Printf("completed: %d / %d (workers: %d)\n", totalCompleted, TotalSeqCnt, numWorkers)
	fmt.Printf("π ≈ %.10f\n", (totalRatio/float64(totalCompleted))*4)
}
