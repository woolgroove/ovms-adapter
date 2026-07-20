package main

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestMeanPoolRawUsesOnlyUnmaskedTokens(t *testing.T) {
	values := []float32{
		1, 3,
		5, 7,
		100, 100,
	}
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}

	got, err := meanPoolRaw(raw, []int64{1, 1, 0}, 3, 2)
	if err != nil {
		t.Fatalf("meanPoolRaw() error = %v", err)
	}
	want := []float32{3, 5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("meanPoolRaw()[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestMeanPoolRawRejectsWrongSize(t *testing.T) {
	if _, err := meanPoolRaw(make([]byte, 7), []int64{1}, 1, 2); err == nil {
		t.Fatal("meanPoolRaw() accepted an invalid raw tensor size")
	}
}
