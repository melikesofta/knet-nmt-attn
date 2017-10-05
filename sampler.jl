#Given two large files in sequence-per-line format, this program
#creates two files that have "numOfLines" lines randomly sampled
#from both of the source files. It is intended for this program to
#sample smaller data from large parallel language files

#Arguments:
#sourceData1:  Path of first file that is to be sampled from
#sourceData2:  Path of second file that is to be sampled from
#outFile1:     Name of the first output file
#outFile2:     Name of the second output file
#numOfLines:   Number of lines for the output files
#maxLength:    Word amount of longest sentence in the sample

#Requirements: Knet, FileIO, StatsBase

for p in ("Knet","FileIO","StatsBase")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, FileIO
using StatsBase: RandIntSampler, randi

function sampleData(sourceData1, sourceData2; outFile1="sampledData1", outFile2="sampledData2", numOfLines=1000, maxLength=70)

  sFile1 = open(sourceData1, "r")
  sFile2 = open(sourceData2, "r")
  tFile1 = open(outFile1, "w")
  tFile2 = open(outFile2, "w")

  sourceArr1 = readlines(sFile1)
  sourceArr2 = readlines(sFile2)
  close(sFile1)
  close(sFile2)

  targetArr1 = Array{AbstractString}(numOfLines)
  targetArr2 = Array{AbstractString}(numOfLines)

  # Draw length(x) elements from a and write them to a pre-allocated array x.
  sample!(sourceArr1, sourceArr2, targetArr1, targetArr2, maxLength)

  for l1 in targetArr1
      write(tFile1, l1)
  end
  for l2 in targetArr2
      write(tFile2, l2)
  end
  close(tFile1)
  close(tFile2)
end

function sample!(a1, a2, x1, x2, maxLength)
    n = min(length(a1), length(a2))
    k = length(x1)
    k == 0 && return x
    k <= n || error("Cannot draw more samples without replacement.")
    if n < k * 24
      fisher_yates_sample!(a1, a2, x1, x2, maxLength)
    else
      self_avoid_sample!(a1, a2, x1, x2, maxLength)
    end
  return x1, x2
end

function fisher_yates_sample!(a1, a2, x1, x2, maxLength)
    n = min(length(a1), length(a2))
    k = length(x1)
    k <= n || error("length(x) should not exceed length(a)")

    inds = Array(Int, n)
    for i = 1:n
      @inbounds inds[i] = i
    end
    @inbounds for i = 1:k
        j = randi(i, n)
        t = inds[j]
        while length(split(a1[t]))>maxLength
          j = randi(i, n)
          t = inds[j]
        end
        inds[j] = inds[i]
        inds[i] = t
        x1[i] = string(a1[t])
        x2[i] = string(a2[t])
    end
    return x1, x2
end

function self_avoid_sample!(a1, a2, x1, x2, maxLength)
  n = min(length(a1), length(a2))
  k = length(x1)
  k <= n || error("length(x) should not exceed length(a)")

  s = Set{Int}()
  sizehint!(s, k)
  rgen = RandIntSampler(n)

  # first one
  idx = rand(rgen)
  while length(split(a1[idx]))>maxLength
    idx = rand(rgen)
  end
  x1[1] = a1[idx]
  x2[1] = a2[idx]
  push!(s, idx)

  # remaining
  for i = 2:k
      idx = rand(rgen)
      while (idx in s || length(split(a1[idx]))>maxLength)
          idx = rand(rgen)
      end
      x1[i] = string(a1[idx])
      x2[i] = string(a2[idx])
      push!(s, idx)
  end
  return x1, x2
end