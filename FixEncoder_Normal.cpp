#include <atomic>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <string_view>
#include <system_error>

using namespace std;

struct AllocationCounters {
   uint64_t scalarNew;
   uint64_t arrayNew;
};

static std::atomic<uint64_t> scalarNewCount{ 0 };
static std::atomic<uint64_t> arrayNewCount{ 0 };

[[nodiscard]] inline AllocationCounters allocationCounters() {
   return {
      scalarNewCount.load( std::memory_order_relaxed ),
      arrayNewCount.load( std::memory_order_relaxed ),
   };
}

void* operator new( std::size_t size ) {
   scalarNewCount.fetch_add( 1, std::memory_order_relaxed );
   void* ptr = std::malloc( size );
   if ( !ptr ) throw std::bad_alloc();
   return ptr;
}

void* operator new[]( std::size_t size ) {
   arrayNewCount.fetch_add( 1, std::memory_order_relaxed );
   void* ptr = std::malloc( size );
   if ( !ptr ) throw std::bad_alloc();
   return ptr;
}

void operator delete( void* ptr ) noexcept {
   std::free( ptr );
}

void operator delete[]( void* ptr ) noexcept {
   std::free( ptr );
}

void operator delete( void* ptr, std::size_t ) noexcept {
   std::free( ptr );
}

void operator delete[]( void* ptr, std::size_t ) noexcept {
   std::free( ptr );
}

struct NewOrderSingleInput {
   uint32_t seq_num;
   std::string_view sender_comp_id;
   std::string_view target_comp_id;
   std::string_view sending_time;
   std::string_view cl_ord_id;
   std::string_view symbol;
   char side;
   std::string_view transact_time;
   uint32_t order_qty;
   char ord_type;
   double price;
   uint8_t price_precision;
   char time_in_force;
};

struct EncodedFixMessage {
   char *begin;
   char *end;
};

// This baseline keeps the same fixed header layout as the optimized encoder.
static constexpr uint32_t FIX44_MIN_BODY_LENGTH = 100;
static constexpr uint32_t FIX44_MAX_BODY_LENGTH = 999;
static constexpr size_t FIX44_HEADER_BYTES = 16;
static constexpr int LIMIT_PRICE_SCALE = 4;

// Conventional scalar byte-at-a-time checksum: no SIMD instructions.
[[gnu::optimize( "no-tree-vectorize" )]]
[[nodiscard]] inline uint32_t fixChecksumScalar( const char *buf, size_t len ) {
   uint32_t sum = 0;
   for ( size_t i = 0; i < len; ++i ) {
      sum += static_cast<uint8_t>( buf[ i ] );
   }
   return sum & 0xff;
}

// Standard-library conversions still write directly to the final FIX buffer.
[[gnu::always_inline]] inline char *writeIntNormal( char *buf, int value ) {
   const auto [ end, error ] = std::to_chars( buf, buf + 11, value );
   assert( error == std::errc{} );
   return end;
}

[[gnu::always_inline]] inline char *writeDoubleNormal(
   char *buf, double value ) {
   // A price has more than enough space in the encoder's fixed packet buffer.
   const auto [ end, error ] = std::to_chars(
      buf, buf + 32, value, std::chars_format::fixed, LIMIT_PRICE_SCALE );
   assert( error == std::errc{} );
   return end;
}

[[gnu::always_inline]] inline char *tagVal(
   char *buf, int tag, int value ) {
   buf = writeIntNormal( buf, tag );
   *buf++ = '=';
   buf = writeIntNormal( buf, value );
   *buf++ = '\001';
   return buf;
}

[[gnu::always_inline]] inline char *tagVal(
   char *buf, int tag, double value ) {
   buf = writeIntNormal( buf, tag );
   *buf++ = '=';
   buf = writeDoubleNormal( buf, value );
   *buf++ = '\001';
   return buf;
}

[[gnu::always_inline]] inline char *tagVal(
   char *buf, int tag, std::string_view value ) {
   buf = writeIntNormal( buf, tag );
   *buf++ = '=';
   std::memcpy( buf, value.data(), value.size() );
   buf += value.size();
   *buf++ = '\001';
   return buf;
}

[[gnu::always_inline]] inline char *tagVal( char *buf, int tag, char value ) {
   buf = writeIntNormal( buf, tag );
   *buf++ = '=';
   *buf++ = value;
   *buf++ = '\001';
   return buf;
}

[[gnu::always_inline]] inline char *writeChecksum( char *buf, uint32_t checksum ) {
   buf = writeIntNormal( buf, 10 );
   *buf++ = '=';
   *buf++ = static_cast<char>( '0' + checksum / 100 );
   *buf++ = static_cast<char>( '0' + ( checksum / 10 ) % 10 );
   *buf++ = static_cast<char>( '0' + checksum % 10 );
   *buf++ = '\001';
   return buf;
}

[[gnu::always_inline]] inline void writeFix44HeaderWithBodyLength(
   char *messageStart, uint32_t bodyLength ) {
   std::memcpy( messageStart, "8=FIX.4.4\0019=", 12 );
   const auto [ end, error ] = std::to_chars(
      messageStart + 12, messageStart + 15, bodyLength );
   assert( error == std::errc{} );
   assert( end == messageStart + 15 );
   messageStart[ 15 ] = '\001';
}

class BuyOrder {
   public:
      EncodedFixMessage encodeOrder( NewOrderSingleInput *order,
                                     char *messageStart ) {
         char *bodyStart = messageStart + FIX44_HEADER_BYTES;
         assert( order->price_precision == LIMIT_PRICE_SCALE );
         char *bodyEnd = tagVal( bodyStart, 35, "D" );
         bodyEnd = tagVal( bodyEnd, 49, order->sender_comp_id );
         bodyEnd = tagVal( bodyEnd, 56, order->target_comp_id );
         bodyEnd = tagVal( bodyEnd, 34, static_cast<int>( order->seq_num ) );
         bodyEnd = tagVal( bodyEnd, 52, order->sending_time );
         bodyEnd = tagVal( bodyEnd, 11, order->cl_ord_id );
         bodyEnd = tagVal( bodyEnd, 21, '1' );
         bodyEnd = tagVal( bodyEnd, 55, order->symbol );
         bodyEnd = tagVal( bodyEnd, 54, order->side );
         bodyEnd = tagVal( bodyEnd, 60, order->transact_time );
         bodyEnd = tagVal( bodyEnd, 38, static_cast<int>( order->order_qty ) );
         bodyEnd = tagVal( bodyEnd, 40, order->ord_type );
         bodyEnd = tagVal( bodyEnd, 44, order->price );
         bodyEnd = tagVal( bodyEnd, 59, order->time_in_force );

         const uint32_t bodyLength = bodyEnd - bodyStart;
         const bool bodyLengthIsThreeDigits =
            bodyLength >= FIX44_MIN_BODY_LENGTH &&
            bodyLength <= FIX44_MAX_BODY_LENGTH;
         assert( bodyLengthIsThreeDigits );
         if ( !bodyLengthIsThreeDigits ) [[unlikely]] std::abort();

         writeFix44HeaderWithBodyLength( messageStart, bodyLength );
         const uint32_t checksum = fixChecksumScalar(
            messageStart, bodyEnd - messageStart );
         return { messageStart, writeChecksum( bodyEnd, checksum ) };
      }
};

// Compiler-only benchmark barrier: it does not emit a hardware load.
[[gnu::always_inline]] inline void doNotOptimize( const void *value ) {
   asm volatile( "" : : "g"( value ) : "memory" );
}

void benchmarkBuyOrderEncoding( BuyOrder& fixOrder, NewOrderSingleInput *order ) {
   static constexpr uint32_t iterations = 1'000'000;
   alignas( 32 ) char buf[ 1024 ];

   double limitPrice = 1.2345;
   const uint32_t initialSeqNum = order->seq_num;
   uint64_t encodedBytes = 0;

   const AllocationCounters allocationsBefore = allocationCounters();
   const auto start = std::chrono::steady_clock::now();
   for ( uint32_t iteration = 0; iteration < iterations; ++iteration ) {
      order->price = limitPrice;
      order->seq_num = initialSeqNum + iteration;
      const EncodedFixMessage message = fixOrder.encodeOrder( order, buf );
      doNotOptimize( message.begin );
      encodedBytes += static_cast<uint64_t>( message.end - message.begin );
      limitPrice += 1.0;
   }
   const auto elapsed = std::chrono::steady_clock::now() - start;
   const AllocationCounters allocationsAfter = allocationCounters();
   const bool allocationFree =
      allocationsAfter.scalarNew == allocationsBefore.scalarNew &&
      allocationsAfter.arrayNew == allocationsBefore.arrayNew;
   assert( allocationFree );

   const auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
      elapsed ).count();
   const double messagesPerSecond =
      static_cast<double>( iterations ) * 1'000'000'000.0 / elapsedNs;
   const double nanosecondsPerMessage =
      static_cast<double>( elapsedNs ) / iterations;
   cout << "normal benchmark: " << iterations << " encodes, " << elapsedNs
        << " ns, " << nanosecondsPerMessage << " ns/msg, "
        << messagesPerSecond << " msg/s, " << encodedBytes
        << " bytes, allocation-free=" << allocationFree << endl;
}

int main() {
   NewOrderSingleInput order{
      1,
      "HFT_CLIENT",
      "BROKER",
      "20260809-12:34:56.789",
      "HFT-000001",
      "AAPL",
      '1',
      "20260809-12:34:56.789",
      100,
      '2',
      1.2345,
      4,
      '0',
   };
   BuyOrder buy;
   alignas( 32 ) char buf[ 1024 ];
   const EncodedFixMessage message = buy.encodeOrder( &order, buf );
   for ( char *ptr = message.begin; ptr != message.end; ++ptr ) {
      cout << ( *ptr == '\001' ? '|' : *ptr );
   }
   cout << endl;

   benchmarkBuyOrderEncoding( buy, &order );
}
