#include <bit>
#include <cassert>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <immintrin.h>
#include <new>
#include <string_view>

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

void* operator new(std::size_t size) {
   scalarNewCount.fetch_add( 1, std::memory_order_relaxed );
   void* ptr = std::malloc( size );
   if ( !ptr ) throw std::bad_alloc();
   return ptr;
}

void* operator new[](std::size_t size) {
   arrayNewCount.fetch_add( 1, std::memory_order_relaxed );
   void* ptr = std::malloc( size );
   if ( !ptr ) throw std::bad_alloc();
   return ptr;
}

void operator delete(void* ptr) noexcept {
   std::free( ptr );
}

void operator delete[](void* ptr) noexcept {
   std::free( ptr );
}

void operator delete(void* ptr, std::size_t) noexcept {
   std::free( ptr );
}

void operator delete[](void* ptr, std::size_t) noexcept {
   std::free( ptr );
}

// 64-byte aligned cache friendly lookup tables
alignas(64) static constexpr uint32_t POW10_32[] = {
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000
};

alignas(64) static constexpr uint64_t POW10_64[] = {
    1ULL, 10ULL, 100ULL, 1000ULL, 10000ULL, 100000ULL, 1000000ULL, 10000000ULL,
    100000000ULL, 1000000000ULL, 10000000000ULL, 100000000000ULL, 1000000000000ULL,
    10000000000000ULL, 100000000000000ULL, 1000000000000000ULL, 10000000000000000ULL,
    100000000000000000ULL, 1000000000000000000ULL, 10000000000000000000ULL
};

static constexpr double POW10[] = {
    1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0
};

alignas(64) static constexpr char DIGITS_100[200] = {
    '0','0', '0','1', '0','2', '0','3', '0','4', '0','5', '0','6', '0','7', '0','8', '0','9',
    '1','0', '1','1', '1','2', '1','3', '1','4', '1','5', '1','6', '1','7', '1','8', '1','9',
    '2','0', '2','1', '2','2', '2','3', '2','4', '2','5', '2','6', '2','7', '2','8', '2','9',
    '3','0', '3','1', '3','2', '3','3', '3','4', '3','5', '3','6', '3','7', '3','8', '3','9',
    '4','0', '4','1', '4','2', '4','3', '4','4', '4','5', '4','6', '4','7', '4','8', '4','9',
    '5','0', '5','1', '5','2', '5','3', '5','4', '5','5', '5','6', '5','7', '5','8', '5','9',
    '6','0', '6','1', '6','2', '6','3', '6','4', '6','5', '6','6', '6','7', '6','8', '6','9',
    '7','0', '7','1', '7','2', '7','3', '7','4', '7','5', '7','6', '7','7', '7','8', '7','9',
    '8','0', '8','1', '8','2', '8','3', '8','4', '8','5', '8','6', '8','7', '8','8', '8','9',
    '9','0', '9','1', '9','2', '9','3', '9','4', '9','5', '9','6', '9','7', '9','8', '9','9'
};

struct NewOrderSingleInput {
    uint32_t seq_num;
    std::string_view sender_comp_id;
    std::string_view target_comp_id;
    std::string_view sending_time;
    std::string_view cl_ord_id;
    std::string_view symbol;
    char side;                   // '1' = Buy, '2' = Sell
    std::string_view transact_time;
    uint32_t order_qty;
    char ord_type;               // '2' = Limit
    double price;
    uint8_t price_precision;
    char time_in_force;          // '0' = Day
};

struct EncodedFixMessage {
    char *begin;
    char *end;
};

// This encoder's NewOrderSingle layout is constrained to a canonical,
// three-digit BodyLength.  8=FIX.4.4<SOH> is 10 bytes and 9=ddd<SOH> is 6.
static constexpr uint32_t FIX44_MIN_BODY_LENGTH = 100;
static constexpr uint32_t FIX44_MAX_BODY_LENGTH = 999;
static constexpr size_t FIX44_HEADER_BYTES = 16;
static constexpr int LIMIT_PRICE_SCALE = 4;

static_assert( FIX44_MIN_BODY_LENGTH == POW10_32[ 2 ] );
static_assert( FIX44_MAX_BODY_LENGTH == POW10_32[ 3 ] - 1 );


// Sum an unaligned prefix one byte at a time.  buf + prefix is then 32-byte
// aligned, so every following AVX2 load can use the aligned load instruction.
[[nodiscard]] inline uint32_t fast_fix_checksum_avx2(const char* buf, size_t len) {
    __m256i zero = _mm256_setzero_si256();
    __m256i acc  = _mm256_setzero_si256();

    uint64_t total_sum = 0;
    size_t i = 0;
    const uintptr_t address = reinterpret_cast<uintptr_t>( buf );
    const size_t prefix = ( 32 - ( address & 31 ) ) & 31;

    for ( ; i < prefix && i < len; ++i ) {
        total_sum += static_cast<uint8_t>( buf[ i ] );
    }

    // buf + i is 32-byte aligned here, and remains aligned as i increases.
    for (; i + 32 <= len; i += 32) {
        __m256i data = _mm256_load_si256(
           reinterpret_cast<const __m256i*>( buf + i ) );
        // Compute sum of unsigned bytes against 0
        __m256i sad  = _mm256_sad_epu8(data, zero);
        acc          = _mm256_add_epi64(acc, sad);
    }

    // Accumulate SIMD register lanes
    total_sum += _mm256_extract_epi64(acc, 0) +
                 _mm256_extract_epi64(acc, 1) +
                 _mm256_extract_epi64(acc, 2) +
                 _mm256_extract_epi64(acc, 3);

    // Process remaining tail bytes (less than 32 bytes)
    for (; i < len; ++i) {
        total_sum += static_cast<uint8_t>(buf[i]);
    }

    return static_cast<uint32_t>(total_sum % 256);
}




[[nodiscard]] inline uint32_t fix_val_bytes(uint32_t val) {
    const uint32_t bits = 32 - std::countl_zero(val | 1);

    // 2. Estimate decimal digits: log10(2) ≈ 0.3010 .... = 1233 / 4096
    const uint32_t digits = (bits * 1233) >> 12;

    // 3. Constant-time adjustment (0 branches)
    return digits + ((val | 1) >= POW10_32[digits]);
}

// Count bytes for 64-bit values (e.g. Timestamps, Order IDs)
[[nodiscard]] inline uint32_t fix_val_bytes64(uint64_t val) {
    const uint32_t bits = 64 - std::countl_zero(val | 1ULL);
    const uint32_t digits = (bits * 1233) >> 12;
    return digits + ((val | 1ULL) >= POW10_64[digits]);
}

// The venue's limit-price scale is fixed at four decimal places.  The input
// remains a double, but this avoids a runtime precision branch and loop.
inline char* fast_dtoa_fixed4_direct(double val, char* buf) {
    auto whole = static_cast<uint64_t>(val);
    double frac = (val - static_cast<double>(whole)) * 10000.0;
    auto frac_int = static_cast<uint64_t>(frac + 0.5);

    if (frac_int >= 10000) {
        whole += 1;
        frac_int = 0;
    }

    const uint32_t w_len = fix_val_bytes64(whole);
    char* w_ptr = buf + w_len;
    buf += w_len;

    uint64_t w_val = whole;
    while (w_val >= 100) {
        const uint64_t q = w_val / 100;
        const uint64_t r = w_val % 100;
        w_val = q;
        w_ptr -= 2;
        std::memcpy(w_ptr, &DIGITS_100[r * 2], 2);
    }
    if (w_val < 10) {
        *--w_ptr = static_cast<char>('0' + w_val);
    } else {
        w_ptr -= 2;
        std::memcpy(w_ptr, &DIGITS_100[w_val * 2], 2);
    }

    *buf++ = '.';
    std::memcpy( buf, &DIGITS_100[ ( frac_int / 100 ) * 2 ], 2 );
    std::memcpy( buf + 2, &DIGITS_100[ ( frac_int % 100 ) * 2 ], 2 );
    return buf + LIMIT_PRICE_SCALE;
}



template <class OrderType>
class Fix4Encoder {
   public:
      EncodedFixMessage encodeOrder( NewOrderSingleInput *order,
                                     char *bodyStart ) {
         return static_cast<OrderType*>( this )->encodeOrderImpl(
            order, bodyStart );
      }
};


[[gnu::always_inline]] inline char *writeUint32( char *buf, uint32_t val ) {
    const uint32_t len = fix_val_bytes( val );
    char *ptr = buf + len;

    while ( val >= 100 ) {
        const uint32_t quotient = val / 100;
        const uint32_t remainder = val % 100;
        val = quotient;
        ptr -= 2;
        std::memcpy( ptr, &DIGITS_100[remainder * 2], 2 );
    }

    if ( val < 10 ) {
        *--ptr = static_cast<char>( '0' + val );
    } else {
        ptr -= 2;
        std::memcpy( ptr, &DIGITS_100[val * 2], 2 );
    }
    return buf + len;
}

[[gnu::always_inline]] inline char *writeInt( char *buf, int val ) {
    if ( val < 0 ) {
        *buf++ = '-';
        return writeUint32( buf, 0 - static_cast<uint32_t>( val ) );
    }
    return writeUint32( buf, static_cast<uint32_t>( val ) );
}

template <int Tag>
[[gnu::always_inline]] inline char *writeTagPrefix( char *buf ) {
    static_assert( Tag >= 10 && Tag <= 99 );
    *buf++ = static_cast<char>( '0' + Tag / 10 );
    *buf++ = static_cast<char>( '0' + Tag % 10 );
    *buf++ = '=';
    return buf;
}

template <int Tag>
[[gnu::always_inline]] inline char *tagVal( char *buf, int val ) {
    buf = writeTagPrefix<Tag>( buf );
    buf = writeInt( buf, val );
    *buf++ = '\001';
    return buf;
}

template <int Tag>
[[gnu::always_inline]] inline char *tagVal( char *buf, uint32_t val ) {
    buf = writeTagPrefix<Tag>( buf );
    buf = writeUint32( buf, val );
    *buf++ = '\001';
    return buf;
}

template <int Tag>
[[gnu::always_inline]] inline char *tagVal( char *buf, double val ) {
    buf = writeTagPrefix<Tag>( buf );
    buf = fast_dtoa_fixed4_direct( val, buf );
    *buf++ = '\001';
    return buf;
}

template <int Tag>
[[gnu::always_inline]] inline char *tagVal( char *buf, std::string_view val ) {
    buf = writeTagPrefix<Tag>( buf );
    std::memcpy( buf, val.data(), val.size() );
    buf += val.size();
    *buf++ = '\001';
    return buf;
}

template <int Tag>
[[gnu::always_inline]] inline char *tagVal( char *buf, char val ) {
    buf = writeTagPrefix<Tag>( buf );
    *buf++ = val;
    *buf++ = '\001';
    return buf;
}

[[gnu::always_inline]] inline char *writeChecksum( char *buf, uint32_t checksum ) {
    *buf++ = '1';
    *buf++ = '0';
    *buf++ = '=';
    *buf++ = static_cast<char>( '0' + checksum / 100 );
    std::memcpy( buf, &DIGITS_100[ ( checksum % 100 ) * 2 ], 2 );
    buf += 2;
    *buf++ = '\001';
    return buf;
}

[[gnu::always_inline]] inline void writeFix44HeaderWithBodyLength(
   char *messageStart, uint32_t bodyLength ) {
   // The caller has already enforced 100 <= BodyLength <= 999, therefore
   // these are three ordinary decimal digits rather than zero padding.
   std::memcpy( messageStart, "8=FIX.4.4\0019=", 12 );
   messageStart[ 12 ] = static_cast<char>( '0' + bodyLength / 100 );
   const uint32_t finalTwoDigits = bodyLength % 100;
   std::memcpy( messageStart + 13, &DIGITS_100[ finalTwoDigits * 2 ], 2 );
   messageStart[ 15 ] = '\001';
}


class BuyOrder : public Fix4Encoder <BuyOrder >{
       public:
      EncodedFixMessage encodeOrderImpl( NewOrderSingleInput *order,
                                         char *messageStart ) {
         // FIX begins at the exact caller-provided packet offset.  Since tag
         // 9 is always three digits, tag 35 has a fixed offset of 16 bytes.
         char *bodyStart = messageStart + FIX44_HEADER_BYTES;

         // BodyLength starts at 35 and ends at this final SOH, before tag 10.
         assert( order->price_precision == LIMIT_PRICE_SCALE );
         char *bodyEnd = tagVal<35>( bodyStart, "D" );
         bodyEnd = tagVal<49>( bodyEnd, order->sender_comp_id );
         bodyEnd = tagVal<56>( bodyEnd, order->target_comp_id );
         bodyEnd = tagVal<34>( bodyEnd, order->seq_num );
         bodyEnd = tagVal<52>( bodyEnd, order->sending_time );
         bodyEnd = tagVal<11>( bodyEnd, order->cl_ord_id );
         bodyEnd = tagVal<21>( bodyEnd, '1' );
         bodyEnd = tagVal<55>( bodyEnd, order->symbol );
         bodyEnd = tagVal<54>( bodyEnd, order->side );
         bodyEnd = tagVal<60>( bodyEnd, order->transact_time );
         bodyEnd = tagVal<38>( bodyEnd, order->order_qty );
         bodyEnd = tagVal<40>( bodyEnd, order->ord_type );
         bodyEnd = tagVal<44>( bodyEnd, order->price );
         bodyEnd = tagVal<59>( bodyEnd, order->time_in_force );

         const uint32_t bodyLength = bodyEnd - bodyStart;
         const bool bodyLengthIsThreeDigits =
            bodyLength >= FIX44_MIN_BODY_LENGTH &&
            bodyLength <= FIX44_MAX_BODY_LENGTH;
         assert( bodyLengthIsThreeDigits );
         if ( !bodyLengthIsThreeDigits ) [[unlikely]] {
            std::abort();
         }

         writeFix44HeaderWithBodyLength( messageStart, bodyLength );

         // Checksum covers the message through the SOH immediately before tag 10.
         const uint32_t checksum = fast_fix_checksum_avx2(
            messageStart, bodyEnd - messageStart );
         return { messageStart, writeChecksum( bodyEnd, checksum ) };
      }
};

class SellOrder : public Fix4Encoder<SellOrder> {

       public:
      EncodedFixMessage encodeOrderImpl( NewOrderSingleInput *, char *bodyStart ) {
         return { bodyStart, bodyStart };
      }


};

// GCC/Clang compiler barrier for benchmark-only observable results.  It emits
// no hardware instruction or load, but the memory clobber keeps prior writes
// to the wire image observable to the compiler.
[[gnu::always_inline]] inline void doNotOptimize( const void *value ) {
   asm volatile( "" : : "g"( value ) : "memory" );
}

void benchmarkBuyOrderEncoding( Fix4Encoder<BuyOrder>& fixOrder,
                                NewOrderSingleInput *order ) {
   static constexpr uint32_t iterations = 1'000'000;
   alignas( 32 ) char buf[ 1024 ];

   double limitPrice = 1.2345;
   const uint32_t initialSeqNum = order->seq_num;
   uint64_t encodedBytes = 0;

   const AllocationCounters allocationsBefore = allocationCounters();
   const auto start = std::chrono::steady_clock::now();
   for ( uint32_t iteration = 0; iteration < iterations; ++iteration ) {
      // The same input object is updated for every new wire message.
      // encodeOrder() always begins at buf; it never appends.
      order->price = limitPrice;
      order->seq_num = initialSeqNum + iteration;
      const EncodedFixMessage message = fixOrder.encodeOrder( order, buf );

      // The compiler must retain the preceding message stores, but no wire
      // byte is read solely for the benchmark.
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

   cout << "benchmark: " << iterations << " encodes, " << elapsedNs
        << " ns, " << nanosecondsPerMessage << " ns/msg, "
        << messagesPerSecond << " msg/s, "
        << encodedBytes << " bytes, allocation-free=" << allocationFree << endl;
}



int  main() {
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
   Fix4Encoder<BuyOrder>& fixOrder = buy;
   EncodedFixMessage message = fixOrder.encodeOrder(
      &order, buf );

   for ( char *ptr = message.begin; ptr != message.end; ++ptr ) {
      cout << ( *ptr == '\001' ? '|' : *ptr );
   }
   cout << endl;

   benchmarkBuyOrderEncoding( fixOrder, &order );
}
