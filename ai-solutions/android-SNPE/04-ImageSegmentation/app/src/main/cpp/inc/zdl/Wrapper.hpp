//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#define SNPE_WRAPPER_TYPES

#include <utility>
#include <memory>
#include <type_traits>
#include <set>

#include <cstddef>

#include <string>


#include "DlSystem/DlError.h"

// Put type aliases in zdl::namespace
#define ALIAS_IN_ZDL_NAMESPACE(ns, type) namespace zdl{ namespace ns { using type = ::ns::type; }}


// Uncomment to print info from the Wrapper base class
//#define WRAPPER_DEBUG_PRINTS


#ifdef WRAPPER_DEBUG_PRINTS

#ifdef _MSC_VER
#define WRAPPER_FUNCTION_NAME __FUNCTION__
#define WRAPPER_TRACE() std::cout << __LINE__ << ":\t" << WRAPPER_FUNCTION_NAME << std::endl
#define WRAPPER_ETRACE() std::cout << __LINE__ << ":\t" << WRAPPER_FUNCTION_NAME << std::endl
#else
#define WRAPPER_FUNCTION_NAME __PRETTY_FUNCTION__
#define WRAPPER_TRACE() std::cout << "\e[33m" << __LINE__ << ":\t" << WRAPPER_FUNCTION_NAME << "\e[0m" << std::endl
#define WRAPPER_ETRACE() std::cout << "\e[31m" << __LINE__ << ":\t" << WRAPPER_FUNCTION_NAME << "\e[0m" << std::endl
#endif

#include <iostream>
#else
#define WRAPPER_TRACE() do{}while(0)
#define WRAPPER_ETRACE() do{}while(0)
#endif


namespace WrapperDetail {


template<typename HandleType, typename MemberType>
using GetterFuncType = MemberType(*)(HandleType);

template<typename HandleType, typename MemberType>
using SetterFuncType = Snpe_ErrorCode_t(*)(HandleType, MemberType);



// Allow Wrappers to have members that require CAPI calls for access
template<typename OwnerType,
         typename HandleType,
         typename MemberType,
         GetterFuncType<HandleType, MemberType> GetterFunc,
         SetterFuncType<HandleType, MemberType> SetterFunc
>
class GenericMemberReference{
  OwnerType& owner;
public:


  ~GenericMemberReference() = default;
  GenericMemberReference() = delete;

  GenericMemberReference(const GenericMemberReference&) = delete;
  GenericMemberReference(GenericMemberReference&&) noexcept = default;

  GenericMemberReference(OwnerType& owner)
    : owner{owner}
  {  }
  explicit GenericMemberReference(OwnerType& owner, MemberType member)
    : owner{owner}
  {
    operator=(member);
  }
  GenericMemberReference& operator=(MemberType member){
    SetterFunc(owner.handle(), member);
    return *this;
  }

  operator MemberType() const{
    return GetterFunc(owner.handle());
  }

  GenericMemberReference&
  operator=(const GenericMemberReference& other){
    return operator=(other.operator MemberType());
  }

  MemberType operator()() const{
    return operator MemberType();
  }

};

// Allow Wrappers to have members that require CAPI calls for access
template<typename OwnerType,
         typename HandleType,
         typename MemberType,
         GetterFuncType<HandleType, MemberType> GetterFunc
>
class GenericConstMemberReference{

  OwnerType& owner;

public:
  ~GenericConstMemberReference() = default;
  GenericConstMemberReference() = delete;

  GenericConstMemberReference(const GenericConstMemberReference&) = delete;
  GenericConstMemberReference(GenericConstMemberReference&&) noexcept = default;

  GenericConstMemberReference(OwnerType& owner)
  : owner{owner}
  {  }

  operator MemberType() const{
    return GetterFunc(owner.handle());
  }


  template<typename T = MemberType, typename std::enable_if<std::is_same<T,std::string>::value,int>::Type=0>
  operator const char*() const{
    thread_local std::string tlss;
    tlss = operator MemberType();
    return tlss.c_str();
  }

  MemberType operator()() const{
    return operator MemberType();
  }

};



// Allows returning references to literals through the CAPI's _Get and _Set functions
template<typename HandleType, typename MemberType, typename IndexType>
using GetterIndexedFuncType = MemberType(*)(HandleType, IndexType);

template<typename HandleType, typename MemberType, typename IndexType>
using SetterIndexedFuncType = Snpe_ErrorCode_t(*)(HandleType, IndexType, MemberType);

template<typename OwnerType,
         typename HandleType,
         typename MemberType,
         typename IndexType,
         GetterIndexedFuncType<HandleType, MemberType, IndexType> GetterFunc,
         SetterIndexedFuncType<HandleType, MemberType, IndexType> SetterFunc
>
class MemberIndexedReference{
  OwnerType& owner;
  IndexType idx;

public:
  MemberIndexedReference(OwnerType& owner, IndexType idx)
    : owner{owner},
      idx{idx}
  {  }
  MemberIndexedReference(const MemberIndexedReference&) noexcept = default;
  MemberIndexedReference(MemberIndexedReference&&) noexcept = default;

  MemberIndexedReference& operator=(const MemberIndexedReference&) noexcept = default;
  MemberIndexedReference& operator=(MemberIndexedReference&&) noexcept = default;

  MemberIndexedReference operator=(MemberType member){
    SetterFunc(owner.handle(), idx, member);
    return *this;
  }

  operator MemberType() const{
    return GetterFunc(owner.handle(), idx);
  }

};



// Allow moving ownership of handles
template<typename Handle>
struct HandleMover {
  Handle handle;
  bool isReference;
};

template<typename Handle>
HandleMover<Handle> moveHandle(Handle handle, bool isReference = false){
  return {handle, isReference};
}

// Virtual base class to allow for WrapperStorage to hold pointers to any Wrapper type
class WrapperBase{
public:
  virtual ~WrapperBase() = default;
};

// Storage type for Wrappers. Will have a set if the CAPI type is capable of creating reference handles
template<typename Handle, bool CreatesRefs>
struct WrapperStorage{
  Handle handle;
  bool isReference;
  constexpr WrapperStorage(Handle handle = {}, bool isReference = false) noexcept
    : handle{handle},
      isReference{isReference}
  {  }
};

template<typename Handle>
struct WrapperStorage<Handle, true>{
  Handle handle;
  bool isReference;
  mutable std::set<std::unique_ptr<WrapperBase>> referencedObjects;
  WrapperStorage(Handle handle = {}, bool isReference = false) noexcept
    : handle{handle},
      isReference{isReference}
  {  }
};

// Allow a handle to be unbound from a Wrapper
struct HandleReleaser{
  template<typename WrapperType>
  static typename WrapperType::HandleType release(WrapperType& wrapper){
    auto toret = wrapper.m_Storage.handle;
    wrapper.m_Storage.handle = {};
    return toret;
  }
};

} // ns WrapperDetail



// The base class for all Wrappers around the CAPI
// NOTE: This Wrapper class leverages the Curiously Recurring Template Pattern (CRTP)
template<typename Derived, typename Handle, bool CreatesRefs = false>
class Wrapper : public WrapperDetail::WrapperBase{
  friend struct WrapperDetail::HandleReleaser;
  // Allow certain types to access getHandle() and handle()
  template<typename, typename, bool>
  friend class Wrapper;

  template<typename, typename H, typename M, typename I,
    WrapperDetail::GetterIndexedFuncType<H,M,I>,
    WrapperDetail::SetterIndexedFuncType<H,M,I>>
  friend class WrapperDetail::MemberIndexedReference;

  template<typename, typename H, typename M, WrapperDetail::GetterFuncType<H,M>>
  friend class WrapperDetail::GenericConstMemberReference;

  template<typename, typename H, typename M, WrapperDetail::GetterFuncType<H,M>, WrapperDetail::SetterFuncType<H,M>>
  friend class WrapperDetail::GenericMemberReference;



protected:
  using HandleType = Handle;
  using BaseType = Wrapper<Derived, HandleType, CreatesRefs>;
  using DeleteFunctionType = Snpe_ErrorCode_t(*)(Handle);

  using StorageType = WrapperDetail::WrapperStorage<HandleType, CreatesRefs>;


  template<typename CapiValueType, typename WrapperValueType,  WrapperDetail::GetterFuncType<HandleType, CapiValueType> Getter>
  static WrapperValueType CastingGetter(HandleType handle){
    return static_cast<WrapperValueType>(Getter(handle));
  }
  template<typename CapiValueType, typename WrapperValueType,  WrapperDetail::SetterFuncType<HandleType, CapiValueType> Setter>
  static Snpe_ErrorCode_t CastingSetter(HandleType handle, WrapperValueType value){
    return Setter(handle,static_cast<CapiValueType>(value));
  }


  template<typename RlType, typename RlHandleType, void* (*Getter)(HandleType), Snpe_ErrorCode_t (*Setter)(HandleType, RlHandleType)>
  struct WrapperMemberReference{
    Derived& owner;

    WrapperMemberReference(Derived& owner)
      : owner{owner}
    {  }
    WrapperMemberReference(Derived& owner, const RlType& other)
      : owner{owner}
    {
      operator=(other);
    }

    WrapperMemberReference& operator=(const RlType& rl){
      Setter(getHandle(owner), getHandle(rl));
      return *this;
    }

    operator RlType&() {
      return *owner.template makeReference<RlType>( Getter(getHandle(owner)) );
    }
    operator RlType&() const {
      return *owner.template makeReference<RlType>( Getter(getHandle(owner)) );
    }

    RlType& operator()(){
      return operator RlType&();
    }
    const RlType& operator()() const{
      return operator RlType&();
    }
  };

  // For Factory/Singleton types, we need a way for the deleter to do nothing
  static Snpe_ErrorCode_t NoOpDeleter(Handle){
    return SNPE_SUCCESS;
  }

  // Simplify calls to WrapperDetail::moveHandle. Can be removed, but will require updating all calls to moveHandle
  template<typename H>
  static WrapperDetail::HandleMover<H> moveHandle(H handle, bool isReference = false){
    return WrapperDetail::moveHandle(handle, isReference);
  }


  HandleType& handle() noexcept{ return m_Storage.handle; }
  const HandleType& handle() const noexcept{ return m_Storage.handle; }

  bool isReference() const noexcept{ return  m_Storage.isReference; }

  void Dtor(){
    if(!isReference() && !handle()){
      if(Derived::DeleteFunction != NoOpDeleter){
        WRAPPER_ETRACE();
      }
    }
    if(!isReference() && handle()){
      WRAPPER_TRACE();
#ifdef WRAPPER_DEBUG_PRINTS
      auto status = Derived::DeleteFunction(handle());
      if(status != SNPE_SUCCESS){
        WRAPPER_ETRACE();
      }
#else
      Derived::DeleteFunction(handle());
#endif

      handle() = nullptr;
    } else {
      WRAPPER_TRACE();
    }
  }

protected:

  // Only compile these if the class creates references. This will save memory and time
  template<bool B = CreatesRefs, typename std::enable_if<B, int>::type=0>
  void addReference(WrapperBase* wrapperBase) const{ // accesses mutable member
    if(!wrapperBase){
      WRAPPER_ETRACE();
    }
    m_Storage.referencedObjects.insert(std::unique_ptr<WrapperBase>(wrapperBase));
  }

  template<typename T, typename H, bool B = CreatesRefs, typename std::enable_if<B, int>::type=0>
  T* makeReference(H referenceHandle) const{
    if(!referenceHandle){
      WRAPPER_ETRACE();
      return nullptr;
    }
    auto refObj = new T(moveHandle(referenceHandle, true));
    addReference(refObj);
    return refObj;
  }

  // This will be used to access another Wrapped type's handles once handle() is made protected
  template<typename OtherDerived, typename OtherHandle, bool OtherCreatesRefs>
  static OtherHandle getHandle(const Wrapper<OtherDerived,OtherHandle,OtherCreatesRefs>& otherObject){
    return otherObject.handle();
  }

  template<typename OtherDerived, typename OtherHandle, bool OtherCreatesRefs>
  static OtherHandle getHandle(const Wrapper<OtherDerived,OtherHandle,OtherCreatesRefs>* otherObject){
    if(!otherObject) return {};
    return getHandle<OtherDerived, OtherHandle, OtherCreatesRefs>(*otherObject);
  }

  template<typename T, typename H>
  static std::unique_ptr<T> makeUnique(H handle){
    if(!handle) return {};
    return std::unique_ptr<T>(new T(moveHandle(handle)));
  }


public:
  ~Wrapper(){
    Dtor();
  }
protected:
  // Only derived types should have access to this
  Wrapper(HandleType handle, bool isReference = false)
  : m_Storage{handle, isReference}
  { WRAPPER_TRACE(); }

public:
  // We should never have an empty wrapper
  Wrapper() = delete;

  // Move semantics are essentially free for all wrapper types
  Wrapper(Wrapper&& other) noexcept
    : m_Storage{std::move(other.m_Storage)}
  {
    WRAPPER_TRACE();
    other.handle() = nullptr;
  }
  Wrapper(const Wrapper&) = delete;


  Wrapper& operator=(Wrapper&& other) noexcept{
    WRAPPER_TRACE();
    if(this != &other){
      std::swap(m_Storage, other.m_Storage);
      other.Dtor();
    }
    return *this;
  }
  Wrapper& operator=(const Wrapper&) = delete;


  // Allow a CAPI handle to be taken over by a Wrapper
  Wrapper(WrapperDetail::HandleMover<HandleType> handleMover) noexcept
  : Wrapper(handleMover.handle, handleMover.isReference)
  { WRAPPER_TRACE(); }

protected:
  // Simplify Derived's move assignment operators
  Derived& moveAssign(Derived&& other) noexcept{ WRAPPER_TRACE();
    return static_cast<Derived&>(operator=(std::move(other)));
  }


private:
  StorageType m_Storage;

};
