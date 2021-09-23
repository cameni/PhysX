//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2021 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

#ifndef PXFOUNDATION_PXPOS_H
#define PXFOUNDATION_PXPOS_H

/** \addtogroup foundation
@{
*/

#include "foundation/PxMath.h"
#include "PxVec3.h"

#if 1
using pfloat = float;
#else
using pfloat = double;
#endif

#if !PX_DOXYGEN
namespace physx
{
#endif

/**
\brief 3 Element vector class.

This is a 3-dimensional vector class with public data members.
*/
class PxPos
{
  public:

	using float_type = pfloat;

	/**
	\brief default constructor leaves data uninitialized.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos()
	{
	}

	/**
	\brief zero constructor.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos(PxZERO r) : x(0.0f), y(0.0f), z(0.0f)
	{
		PX_UNUSED(r);
	}

	/**
	\brief Assigns scalar parameter to all elements.

	Useful to initialize to zero or one.

	\param[in] a Value to assign to elements.
	*/
	explicit PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos(pfloat a) : x(a), y(a), z(a)
	{
	}

	/**
	\brief Initializes from 3 scalar parameters.

	\param[in] nx Value to initialize X component.
	\param[in] ny Value to initialize Y component.
	\param[in] nz Value to initialize Z component.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos(pfloat nx, pfloat ny, pfloat nz) : x(nx), y(ny), z(nz)
	{
	}

	/**
	\brief Copy ctor.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos(const PxPos& v) : x(v.x), y(v.y), z(v.z)
	{
	}

	/**
	\brief Initialize from direction vector and distance.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos(const PxVec3& v, pfloat d) : x(v.x * d), y(v.y * d), z(v.z * d)
	{
	}

	// Operators

	/**
	\brief Assignment operator
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos& operator=(const PxPos& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		return *this;
	}

	/**
	\brief element access
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE pfloat& operator[](unsigned int index)
	{
		PX_SHARED_ASSERT(index <= 2);

		return reinterpret_cast<pfloat*>(this)[index];
	}

	/**
	\brief element access
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE const pfloat& operator[](unsigned int index) const
	{
		PX_SHARED_ASSERT(index <= 2);

		return reinterpret_cast<const pfloat*>(this)[index];
	}

	/**
	\brief returns true if the two vectors are exactly equal.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE bool operator==(const PxPos& v) const
	{
		return x == v.x && y == v.y && z == v.z;
	}

	/**
	\brief returns true if the two vectors are not exactly equal.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE bool operator!=(const PxPos& v) const
	{
		return x != v.x || y != v.y || z != v.z;
	}

	/**
	\brief tests for exact zero vector
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE bool isZero() const
	{
		return x == 0.0f && y == 0.0f && z == 0.0f;
	}

	/**
	\brief returns true if all 3 elems of the vector are finite (not NAN or INF, etc.)
	*/
	PX_CUDA_CALLABLE PX_INLINE bool isFinite() const
	{
		return PxIsFinite(x) && PxIsFinite(y) && PxIsFinite(z);
	}

	/**
	\brief vector addition
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos operator+(const PxVec3& v) const
	{
		return PxPos(x + v.x, y + v.y, z + v.z);
	}

	/**
	\brief vector difference
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos operator-(const PxVec3& v) const
	{
		return PxPos(x - v.x, y - v.y, z - v.z);
	}

	/**
	\brief position difference
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 operator-(const PxPos& v) const
	{
		return PxVec3(float(x - v.x), float(y - v.y), float(z - v.z));
	}

	/**
	\brief vector addition
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos& operator+=(const PxVec3& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	/**
	\brief vector difference
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxPos& operator-=(const PxVec3& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE pfloat dot(const PxVec3& v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}

	/**
	\brief returns MIN(x, y, z);
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE pfloat minElement() const
	{
		return PxMin(x, PxMin(y, z));
	}

	/**
	\brief returns MAX(x, y, z);
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE pfloat maxElement() const
	{
		return PxMax(x, PxMax(y, z));
	}

	pfloat x, y, z;
};

//PX_CUDA_CALLABLE static PX_FORCE_INLINE PxPos operator+(const PxVec3& v, const PxPos& p)
//{
//	return PxPos(p.x + v.x, p.y + v.y, p.z + v.z);
//}


#if !PX_DOXYGEN
} // namespace physx
#endif

/** @} */
#endif // #ifndef PXFOUNDATION_PXPOS_H
