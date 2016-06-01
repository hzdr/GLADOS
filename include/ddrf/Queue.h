#ifndef DDRF_QUEUE_H_
#define DDRF_QUEUE_H_

#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

namespace ddrf
{
	template <class T>
	auto queue_limit(T t) -> std::size_t
	{
		return static_cast<std::size_t>(t);
	}

	template <class Object>
	class Queue
	{
		public:
			/*
			 * The default constructed Queue has no limit, hence the member is 0.
			 */
			Queue() : limit_{0u}, count_{0u} {}
			explicit Queue(std::size_t limit) : limit_{limit}, count_{0u} {}

			/*
			 * Item and Object are of the same type but we need this extra template to make use of the
			 * nice reference collapsing rules
			 */
			template <class Item>
			void push(Item&& item)
			{
				auto lock = std::unique_lock<decltype(mutex_)>{mutex_};
				if(limit_ != 0u)
				{
					while(count_ >= limit_)
						count_cv_.wait(lock);
				}

				queue_.push(std::forward<Item>(item));

				if(limit_ != 0u)
					++count_;

				item_cv_.notify_one();
			}

			Object take()
			{
				auto lock = std::unique_lock<decltype(mutex_)>{mutex_};
				while(queue_.empty())
					item_cv_.wait(lock);

				auto ret = std::move(queue_.front());
				queue_.pop();

				if(limit_ != 0u)
				{
					--count_;
					count_cv_.notify_one();
				}

				return ret;
			}

		private:
			const std::size_t limit_;
			std::size_t count_;
			mutable std::mutex mutex_;
			std::condition_variable item_cv_, count_cv_;
			std::queue<Object> queue_;

	};
}

#endif /* DDRF_QUEUE_H_ */
