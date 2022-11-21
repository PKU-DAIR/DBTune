select current_timestamp(6) into @query_start;
set @query_name='14.sql';
SELECT 100.00 * SUM(CASE WHEN P_TYPE LIKE 'PROMO%' THEN L_EXTENDEDPRICE * (1 - L_DISCOUNT) ELSE 0 END) / SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) AS PROMO_REVENUE FROM LINEITEM, PART WHERE L_PARTKEY = P_PARTKEY AND L_SHIPDATE >= DATE '1996-12-01' AND L_SHIPDATE < DATE '1996-12-01' + INTERVAL '1' MONTH;
set @query_time_ms= timestampdiff(microsecond, @query_start, current_timestamp(6))/1000;
SELECT @query_name, @query_time_ms;